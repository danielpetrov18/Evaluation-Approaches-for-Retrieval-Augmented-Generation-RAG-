class PromptHelper:
    
    def __init__(self, client):
        self.__client = client

    def add_prompt(self, name: str, template: str, input_types: dict):
        return self.__client.add_prompt(name=prompt_name, template=template, input_types=input_types)
        
    def get_all_prompts(self):
        return self.__client.get_all_prompts()['results']
    
    def update_prompt(self, name: str, template: str, input_types: dict):
        return self.__client.update_prompt(name=prompt_name, template=template, input_types=input_types)    
    
    def delete_prompt(self, prompt_name: str):
        return self.__client.delete_prompt(prompt_name)