import requests

def test_api(url):
    try:
        response = requests.get(url, timeout=10)
        print(f"Status code: {response.status_code}")
        print("Response body:")
        print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")

if __name__ == "__main__":
    # Replace with your API endpoint
    test_api("https://agents.aetherraid.dev")