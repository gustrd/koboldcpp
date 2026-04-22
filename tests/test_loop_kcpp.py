import unittest
from unittest.mock import MagicMock, patch, mock_open
import json
import time
import subprocess
import os
import sys

# Add root to path so we can import loop_kcpp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import loop_kcpp

class TestLoopKCPP(unittest.TestCase):

    def test_search_json_for_string(self):
        data = {
            "workers": [
                {"name": "MyWorker1", "id": 1},
                {"name": "OtherWorker", "id": 2}
            ],
            "status": "ok"
        }
        self.assertTrue(loop_kcpp._search_json_for_string(data, "MyWorker1"))
        self.assertTrue(loop_kcpp._search_json_for_string(data, "myworker1"))  # case insensitive
        self.assertTrue(loop_kcpp._search_json_for_string(data, "status"))     # key match
        self.assertFalse(loop_kcpp._search_json_for_string(data, "NonExistent"))

    @patch("urllib.request.urlopen")
    def test_http_contains_found(self, mock_urlopen):
        # Mock response with substring
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"workers": [{"name": "MyWorker"}]}'
        mock_response.headers.get.return_value = "application/json"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        self.assertTrue(loop_kcpp.http_contains("http://test.com", "MyWorker"))

    @patch("urllib.request.urlopen")
    def test_http_contains_not_found(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"workers": [{"name": "Other"}]}'
        mock_response.headers.get.return_value = "application/json"
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        self.assertFalse(loop_kcpp.http_contains("http://test.com", "MyWorker"))

    @patch("urllib.request.urlopen")
    def test_http_contains_error_returns_true(self, mock_urlopen):
        # The logic states: Returns True for errors (assumes worker is still online)
        mock_urlopen.side_effect = Exception("Connection error")
        self.assertTrue(loop_kcpp.http_contains("http://test.com", "MyWorker"))

    def test_interruptible_sleep(self):
        loop_kcpp.stop_event.clear()
        # Test normal completion
        # Use very short sleep for test
        with patch("time.sleep") as mock_sleep:
            interrupted = loop_kcpp.interruptible_sleep(2, 1)
            self.assertFalse(interrupted)
            self.assertEqual(mock_sleep.call_count, 2)

        # Test interruption
        loop_kcpp.stop_event.set()
        interrupted = loop_kcpp.interruptible_sleep(10, 1)
        self.assertTrue(interrupted)
        loop_kcpp.stop_event.clear()

    @patch("psutil.sensors_battery")
    def test_get_battery_percent_psutil(self, mock_battery):
        mock_battery.return_value = MagicMock(percent=75.0)
        self.assertEqual(loop_kcpp.get_battery_percent(), 75)

    @patch("builtins.open", new_callable=mock_open)
    def test_log_to_file(self, mock_file):
        loop_kcpp.log_to_file("Test Message", "test.log")
        mock_file.assert_called_with("test.log", "a", encoding="utf-8")
        handle = mock_file()
        handle.write.assert_called()

    @patch("psutil.Process")
    @patch("psutil.wait_procs")
    def test_ensure_terminate_process(self, mock_wait_procs, mock_process_cls):
        mock_popen = MagicMock(spec=subprocess.Popen)
        mock_popen.pid = 999
        mock_popen.poll.return_value = None # Process is alive

        mock_proc = MagicMock()
        mock_process_cls.return_value = mock_proc
        mock_proc.children.return_value = []
        mock_wait_procs.return_value = ([mock_proc], []) # All gone

        loop_kcpp.ensure_terminate_process(mock_popen)
        mock_proc.terminate.assert_called()

if __name__ == "__main__":
    unittest.main()
