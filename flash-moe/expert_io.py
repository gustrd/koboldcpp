"""
expert_io.py — Phase 2B Module 2

Async Windows file I/O pool for reading expert weight files produced by
extract_experts.py.

All reads use FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED via ctypes so
that data bypasses the OS page cache and comes directly from the SSD.
Buffers are allocated with VirtualAlloc for 4 096-byte sector alignment.

Public API:
    pool = ExpertIOPool(experts_dir, index_json_path)
    pool.submit(layer=0, expert_ids=[3, 7, 12, ...])
    buffers: list[bytes] = pool.wait()          # one bytes per expert_id
    pool.destroy()                               # free VirtualAlloc'd memory

    pool.avg_latency_ms   -> float
    pool.total_bytes_read -> int
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import time
from pathlib import Path
from typing import Sequence

# ── Windows API constants ─────────────────────────────────────────────────────
_GENERIC_READ            = 0x80000000
_FILE_SHARE_READ         = 0x00000001
_OPEN_EXISTING           = 3
_FILE_FLAG_NO_BUFFERING  = 0x20000000
_FILE_FLAG_OVERLAPPED    = 0x40000000
_INVALID_HANDLE_VALUE    = ctypes.c_void_p(-1).value
_INFINITE                = 0xFFFFFFFF
_WAIT_OBJECT_0           = 0
_MEM_COMMIT              = 0x00001000
_MEM_RESERVE             = 0x00002000
_MEM_RELEASE             = 0x00008000
_PAGE_READWRITE          = 0x04

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# Fix return/arg types to avoid 32-bit truncation on 64-bit Windows.
_kernel32.VirtualAlloc.restype  = ctypes.c_void_p
_kernel32.VirtualAlloc.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD,
]
_kernel32.VirtualFree.restype   = wt.BOOL
_kernel32.VirtualFree.argtypes  = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]

_kernel32.CreateFileW.restype   = wt.HANDLE
_kernel32.CreateEventW.restype  = wt.HANDLE

_kernel32.ReadFile.restype  = wt.BOOL
_kernel32.ReadFile.argtypes = [
    wt.HANDLE,                      # hFile
    ctypes.c_void_p,                # lpBuffer
    wt.DWORD,                       # nNumberOfBytesToRead
    ctypes.POINTER(wt.DWORD),       # lpNumberOfBytesRead (may be NULL for OVERLAPPED)
    ctypes.c_void_p,                # lpOverlapped  (void* keeps struct layout simple)
]

_kernel32.GetOverlappedResult.restype  = wt.BOOL
_kernel32.GetOverlappedResult.argtypes = [
    wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(wt.DWORD), wt.BOOL,
]

_kernel32.WaitForSingleObject.restype  = wt.DWORD
_kernel32.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]

# Correct OVERLAPPED for 64-bit Windows.
# Internal/InternalHigh are ULONG_PTR (pointer-sized = 8 bytes on 64-bit).
# Using c_ulong (4 bytes on Windows) gives a 24-byte struct; Windows expects 32.
class _OffsetStruct(ctypes.Structure):
    _fields_ = [("Offset", wt.DWORD), ("OffsetHigh", wt.DWORD)]

class _OffsetUnion(ctypes.Union):
    _fields_ = [("_s", _OffsetStruct), ("Pointer", ctypes.c_void_p)]

class _OVERLAPPED(ctypes.Structure):
    _fields_ = [
        ("Internal",     ctypes.c_size_t),   # ULONG_PTR
        ("InternalHigh", ctypes.c_size_t),   # ULONG_PTR
        ("_union",       _OffsetUnion),
        ("hEvent",       wt.HANDLE),
    ]


def _last_error_msg() -> str:
    return ctypes.FormatError(ctypes.get_last_error())


def _create_event() -> wt.HANDLE:
    h = _kernel32.CreateEventW(None, True, False, None)
    if not h:
        raise OSError(f"CreateEventW failed: {_last_error_msg()}")
    return h


def _close_handle(h) -> None:
    if h and h != _INVALID_HANDLE_VALUE:
        _kernel32.CloseHandle(h)


def _virtual_alloc(size: int) -> int:
    """Allocate *size* bytes of 4K-aligned memory via VirtualAlloc."""
    ptr = _kernel32.VirtualAlloc(None, size, _MEM_COMMIT | _MEM_RESERVE, _PAGE_READWRITE)
    if not ptr:
        raise MemoryError(f"VirtualAlloc({size}) failed: {_last_error_msg()}")
    return ptr


def _virtual_free(ptr: int) -> None:
    if ptr:
        _kernel32.VirtualFree(ctypes.c_void_p(ptr), 0, _MEM_RELEASE)


_SECTOR = 4096


def _align_up(n: int, align: int = _SECTOR) -> int:
    return (n + align - 1) & ~(align - 1)


# ── ExpertIOPool ──────────────────────────────────────────────────────────────

class ExpertIOPool:
    """
    Submits overlapped reads for K expert files and waits for completion.

    submit() is non-blocking; wait() blocks until all K reads are done.
    Buffers are 4 096-byte aligned (VirtualAlloc).

    The pool does NOT write anything — ever.
    """

    def __init__(self, experts_dir: str | Path, index_json_path: str | Path) -> None:
        self._experts_dir = Path(experts_dir)
        self._index       = json.loads(Path(index_json_path).read_text(encoding="utf-8"))

        # Latency / bytes tracking
        self._total_bytes_read: int   = 0
        self._total_latency_ms: float = 0.0
        self._read_count: int         = 0

        # Pending batch state (submit → wait)
        self._pending_layer:    int | None        = None
        self._pending_ids:      list[int]         = []
        self._pending_handles:  list              = []
        self._pending_overlaps: list              = []
        self._pending_buffers:  list[tuple[int,int]] = []  # (ptr, alloc_size)
        self._pending_sizes:    list[int]         = []

    # ── public API ────────────────────────────────────────────────────────────

    def submit(self, layer: int, expert_ids: Sequence[int]) -> None:
        """
        Issue async reads for *expert_ids* in *layer*.  Non-blocking.
        Must be followed by a call to wait() before the next submit().
        """
        if self._pending_handles:
            raise RuntimeError("Previous submit() not yet wait()'d")

        self._pending_layer = layer
        self._pending_ids   = list(expert_ids)

        for exp_id in expert_ids:
            info = self._index["experts"][f"{layer}_{exp_id}"]
            fpath = str(self._experts_dir / info["file"])

            # Sector-aligned read size
            file_size   = info["file_size"]
            read_size   = _align_up(file_size)

            # Allocate aligned buffer
            buf_ptr = _virtual_alloc(read_size)
            self._pending_buffers.append((buf_ptr, read_size))
            self._pending_sizes.append(file_size)

            # Open file (no-buffering, overlapped)
            hfile = _kernel32.CreateFileW(
                fpath,
                _GENERIC_READ,
                _FILE_SHARE_READ,
                None,
                _OPEN_EXISTING,
                _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_OVERLAPPED,
                None,
            )
            if hfile == _INVALID_HANDLE_VALUE:
                raise OSError(f"CreateFileW({fpath}) failed: {_last_error_msg()}")
            self._pending_handles.append(hfile)

            # Overlapped struct with manual-reset event
            ov = _OVERLAPPED()
            ov.hEvent = _create_event()
            self._pending_overlaps.append(ov)

            # Issue the async read (offset 0).
            # Pass buffer as c_void_p and overlapped as its address.
            bytes_read = wt.DWORD(0)
            ok = _kernel32.ReadFile(
                hfile,
                ctypes.c_void_p(buf_ptr),           # aligned buffer
                wt.DWORD(read_size),                 # must be multiple of sector size
                ctypes.byref(bytes_read),            # ignored for OVERLAPPED but required
                ctypes.addressof(ov),                # OVERLAPPED* as integer address
            )
            # ReadFile with OVERLAPPED returns 0 + ERROR_IO_PENDING on success
            last_err = ctypes.get_last_error()
            ERROR_IO_PENDING = 997
            if not ok and last_err != ERROR_IO_PENDING:
                raise OSError(f"ReadFile({fpath}) failed: {_last_error_msg()}")

    def wait(self) -> list[bytes]:
        """
        Wait for all submitted reads to complete and return a list of bytes
        objects (one per expert_id, in submission order).

        Clears the pending batch so submit() can be called again.
        """
        if not self._pending_handles:
            raise RuntimeError("No pending submit()")

        t0 = time.perf_counter()

        results: list[bytes] = []
        try:
            for i, (hfile, ov, (buf_ptr, alloc_size), file_size) in enumerate(
                zip(self._pending_handles, self._pending_overlaps,
                    self._pending_buffers, self._pending_sizes)
            ):
                # Wait for this specific event
                ret = _kernel32.WaitForSingleObject(ov.hEvent, _INFINITE)
                if ret != _WAIT_OBJECT_0:
                    raise OSError(f"WaitForSingleObject failed (ret={ret:#x})")

                # Get actual bytes transferred
                bytes_transferred = wt.DWORD(0)
                ok = _kernel32.GetOverlappedResult(
                    hfile,
                    ctypes.addressof(ov),           # OVERLAPPED* as integer
                    ctypes.byref(bytes_transferred),
                    False,
                )
                if not ok:
                    raise OSError(f"GetOverlappedResult failed: {_last_error_msg()}")

                # Copy the expert data (file_size, not the padded read_size)
                buf_bytes = (ctypes.c_char * file_size).from_address(buf_ptr)
                results.append(bytes(buf_bytes))
                self._total_bytes_read += file_size
        finally:
            self._cleanup_pending()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self._total_latency_ms += elapsed_ms
        self._read_count       += 1

        return results

    def destroy(self) -> None:
        """Free all VirtualAlloc'd memory (safe to call multiple times)."""
        self._cleanup_pending()

    def __del__(self) -> None:
        self.destroy()

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def avg_latency_ms(self) -> float:
        return self._total_latency_ms / self._read_count if self._read_count > 0 else 0.0

    @property
    def total_bytes_read(self) -> int:
        return self._total_bytes_read

    # ── internals ─────────────────────────────────────────────────────────────

    def _cleanup_pending(self) -> None:
        for hfile in self._pending_handles:
            _close_handle(hfile)
        for ov in self._pending_overlaps:
            _close_handle(ov.hEvent)
        for ptr, _ in self._pending_buffers:
            _virtual_free(ptr)

        self._pending_handles  = []
        self._pending_overlaps = []
        self._pending_buffers  = []
        self._pending_sizes    = []
        self._pending_ids      = []
        self._pending_layer    = None
