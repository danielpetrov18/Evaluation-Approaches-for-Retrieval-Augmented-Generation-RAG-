import os 
import csv
import requests
from r2r import R2RClient, R2RException

"""
    This is ran every time the backend client is created. It first tries to register.
    If no such user exists it creates one and then it logs in. Finally, it makes a requests
    to create an index for the vector store. If one already exists it doesn't get overwritten 
    because we use the {'replace': False} option. If a user exists this mean the index should 
    already exist.
    
    https://r2r-docs.sciphi.ai/api-reference/endpoint/create_vector_index
"""

INDEX_URL_SUFFIX = "/v2/create_vector_index"

def get_credentials() -> (str, str):
    with open("./res/user_credentials.csv", mode='r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            email = row[0].strip("'")  
            password = row[1].strip("'") 
    return email, password

def create_vector_index(token: str, url: str) -> None:
    if token is None:
        raise Exception("You must provide a token to create an index!")
    
    body = {
        "table_name": "CHUNKS", 
        "index_method": "ivfflat",
        "measure": "cosine_distance",
        "replace": False,
        "concurrently": True
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    resp = requests.post(url=url, json=body, headers=headers)
    if resp.status_code == 200:
        print(resp.json()['results']['message'])
    else:
        print("Couldn't create an index for vector store!")
        exit(1)

if __name__ == "__main__":
    try:
        email, password = get_credentials()
    except FileNotFoundError as fnfe:
        print(fnfe)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
    
    HOST = os.getenv('R2R_HOSTNAME', 'http://localhost')
    PORT = os.getenv('R2R_PORT', '7272')
    URL = f"{HOST}:{PORT}"
    client = R2RClient(URL)
    
    # https://r2r-docs.sciphi.ai/cookbooks/user-auth#user-profile-management
    try:
        client.register(email, password)    
        token = client.login(email, password)['results']['access_token']['token']
        create_vector_index(token, f"{URL}{INDEX_URL_SUFFIX}")
    except R2RException as r2re:
        #print(r2re)
        exit(1)
    except Exception as e:
        print(e)
        exit(1)