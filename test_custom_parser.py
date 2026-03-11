import re
import json

# ─── Lógica de extração (espelha o que foi implementado no koboldcpp.py) ────────

def normalize_tool_call(obj):
    """Normaliza vários formatos de tool call para o formato OpenAI."""
    if "type" in obj and "function" in obj:
        return obj  # Já está no formato correto
    if "name" in obj and ("arguments" in obj or "parameters" in obj):
        args = obj.get("arguments", obj.get("parameters", {}))
        return {"type": "function", "function": {"name": obj["name"], "arguments": args}}
    if "function" in obj and isinstance(obj["function"], dict):
        func = obj["function"]
        if "name" in func:
            return {"type": "function", "function": {"name": func["name"], "arguments": func.get("arguments", func.get("parameters", {}))}}
    return obj

def extract_json_from_string(text):
    """Extrai os primeiros objetos/arrays JSON válidos do texto."""
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        pass
    try:
        parsed = json.loads(f"[{text}]")
        return parsed
    except Exception:
        pass
    try:
        matches = re.findall(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
        for m in matches:
            try:
                parsed = json.loads(m)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                continue
    except Exception:
        pass
    return []

def extract_all_tool_calls(text):
    """
    Lógica combinada que espelha exatamente o que koboldcpp.py agora executa.
    1. Tenta tag customizada [TOOL_CALLS]nome[ARGS]{json}
    2. Se não encontrar, tenta extrair JSON bruto no formato OpenAI
    Retorna (tool_calls_list, texto_limpo)
    """
    if not text:
        return [], text

    tool_calls = []
    clean_text = text

    # Passo 1: Tag customizada
    custom_pattern = r'\[TOOL_CALLS\](.*?)\[ARGS\](\{.*?\})'
    for match in re.finditer(custom_pattern, text, re.DOTALL):
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        try:
            json.loads(args_str)
            tool_calls.append({"type": "function", "function": {"name": tool_name, "arguments": args_str}})
            clean_text = clean_text.replace(match.group(0), "")
        except Exception:
            pass
    clean_text = clean_text.strip()

    # Passo 2: Fallback para JSON bruto
    if len(tool_calls) == 0:
        tool_calls = extract_json_from_string(clean_text)

    if tool_calls:
        tool_calls = [normalize_tool_call(obj) for obj in tool_calls]
        tool_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name")]  # Filtra objetos inválidos
        for tc in tool_calls:
            tcarg = tc.get("function", {}).get("arguments")
            if not tc.get("id"):
                tc["id"] = "call_12345"  # ID fixo para facilitar o teste
            if tcarg is not None and not isinstance(tcarg, str):
                tc["function"]["arguments"] = json.dumps(tcarg)

    if tool_calls:
        clean_text = clean_text if clean_text else None

    return tool_calls, clean_text


# ─── Casos de Teste ──────────────────────────────────────────────────────────────

def run_test(title, text):
    print(f"\n{'='*60}")
    print(f"TESTE: {title}")
    print(f"{'='*60}")
    print(f"Entrada:\n{text}\n")
    tools, clean = extract_all_tool_calls(text)
    print(f"✅ Tool Calls Extraídos ({len(tools)}):")
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    print(f"\n💬 Texto Limpo Restante:\n{clean}")

if __name__ == "__main__":
    # CASO 1: Tag customizada [TOOL_CALLS]...[ARGS]
    run_test(
        "[TOOL_CALLS] tag format",
        """Algum texto antes.

###
[TOOL_CALLS]message[ARGS]{"action": "send", "channel": "telegram", "to": "7143640756", "message": "Teste"}
###

Texto depois."""
    )

    # CASO 2: JSON bruto no formato OpenAI (seu novo caso)
    run_test(
        "Raw OpenAI JSON array format",
        """[
{
"id": "call_001",
"type": "function",
"function": {
"name": "read",
"arguments": {
"path": "/root/.openclaw/workspace/SOUL.md"
}
}
},
{
"id": "call_001",
"type": "function",
"function": {
"name": "read",
"arguments": {
"path": "/root/.openclaw/workspace/USER.md"
}
}
}
]"""
    )

    # CASO 3: Múltiplas tags [TOOL_CALLS]
    run_test(
        "Multiplas tags [TOOL_CALLS]",
        """[TOOL_CALLS]get_weather[ARGS]{"city": "Sao Paulo"}
[TOOL_CALLS]send_email[ARGS]{"to": "user@example.com", "body": "Hello"}"""
    )

    # CASO 4: Texto puro sem tool call (não deve extrair nada)
    run_test(
        "Texto puro (sem tool call)",
        "Esta é uma resposta normal do modelo sem nenhuma chamada de ferramenta."
    )
