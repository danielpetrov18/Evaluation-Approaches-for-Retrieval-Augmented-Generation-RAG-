import os 
import csv
import requests
from r2r import R2RClient
from r2r import R2RException

INDEX_URL_SUFFIX = "/v2/create_vector_index"

def get_credentials():
    with open("user_credentials.csv", mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            email = row[0].strip("'")  
            password = row[1].strip("'") 
    return email, password

# https://r2r-docs.sciphi.ai/api-reference/endpoint/create_vector_index
def create_vector_index(token: str):
    body = {
        "table_name": "CHUNKS",
        "index_method": "hnsw",
        "measure": "cosine_distance",
        "replace": False,
        "concurrently": True
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    resp = requests.post(f"{URL}{INDEX_URL_SUFFIX}", json=body, headers=headers)
    if resp.status_code == 200:
        print(resp.json()['results']['message'])
    else:
        print("Couldn't create an index for vector store!")
        exit(1)

if __name__ == "__main__":
    email, password = get_credentials()
    
    URL = os.getenv('R2R_HOSTNAME', 'http://localhost:7272')
    client = R2RClient(URL)
    
    # https://r2r-docs.sciphi.ai/cookbooks/user-auth#user-profile-management
    try:
        client.register(email, password)    
        token = client.login(email, password)['results']['access_token']['token']
        create_vector_index(token)
    except R2RException as r2re: # If already registered
        # token = client.login(email, password)['results']['access_token']['token']
        # create_vector_index(token)
        exit(1)
    except Exception as e: # Something unexpected occurs
        print(e)
        exit(1)