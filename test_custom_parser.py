import re
import json

def extract_custom_tool_calls(text):
    """
    Procura por padrões como:
    [TOOL_CALLS]nome_da_ferramenta[ARGS]{"arg1": "valor"}
    
    Retorna uma lista de tool_calls no formato OpenAI e o texto limpo (sem a chamada da ferramenta).
    """
    if not text:
        return [], text

    # Regex para capturar [TOOL_CALLS]nome_da_ferramenta[ARGS]{...}
    # re.DOTALL é importante caso o JSON tenha quebras de linha
    pattern = r'\[TOOL_CALLS\](.*?)\[ARGS\](\{.*?\})'
    
    tool_calls = []
    clean_text = text
    
    # Encontra todas as ocorrências do padrão no texto
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        full_match = match.group(0)
        tool_name = match.group(1).strip()
        args_json_str = match.group(2).strip()
        
        try:
            # Valida se os argumentos são um JSON válido
            args_dict = json.loads(args_json_str)
            
            # Formato estruturado OAI
            tool_calls.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_json_str # OpenAI espera arguments como string JSON
                }
            })
            
            # Remove a chamada de ferramenta inteira do texto final
            clean_text = clean_text.replace(full_match, "")
        except json.JSONDecodeError as e:
            print(f"Erro ao converter argumentos da ferramenta {tool_name} para JSON: {e}")
            # Se for um JSON inválido, ignoramos ou logamos
            pass
            
    return tool_calls, clean_text.strip()


def run_tests():
    sample_text = """Algum texto explicativo gerado pelo LLM.

###
[TOOL_CALLS]message[ARGS]{"action": "send", "channel": "telegram", "to": "7143640756", "message": "Inovação Intel Lunar Lake vs. Panther Lake: Eficiência Energética e Segurança Automotiva\\n\\nO novo processador da Intel..."}
###

Mais algum texto complementar do modelo."""

    print("--- Texto Original ---")
    print(sample_text)
    print("\n----------------------\n")
    
    tools, texto_limpo = extract_custom_tool_calls(sample_text)
    
    print("--- Tool Calls Estruturados (OpenAI Format) ---")
    print(json.dumps(tools, indent=2, ensure_ascii=False))
    
    print("\n--- Texto Limpo (Sem a tag da ferramenta) ---")
    print(texto_limpo)

if __name__ == "__main__":
    run_tests()
