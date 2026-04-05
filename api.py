import requests
import json

def get_dm_response(player_input):
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "dnd-dm",
        "prompt": player_input,
        "format": "json", # This is the magic line
        "stream": False
    }

    response = requests.post(url, json=payload)
    data = json.loads(response.text)
    
    # Now you can parse the AI's "thoughts"
    game_data = json.loads(data['response'])
    
    print(f"DM Says: {game_data['description']}")
    print(f"System Log: HP Change is {game_data['hp_change']}")
    
    return game_data

# Test it
get_dm_response("I swing my longsword at the goblin!")