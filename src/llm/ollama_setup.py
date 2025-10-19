from openai import OpenAI, APIConnectionError


def get_client() -> OpenAI:
    return OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='ollama',
    )

def is_ollama_server_running(client) -> bool:
    try:
        client.models.list() # A lightweight request to check the connection
        return True
    except APIConnectionError:
        print("\n❌ ERROR: Could not connect to the Ollama server.")
        print("Please ensure the Ollama server is running.\n")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred while checking the Ollama server: {e}")
        return False
