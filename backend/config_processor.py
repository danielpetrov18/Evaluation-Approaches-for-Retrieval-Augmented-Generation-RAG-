import os
import tomli
import tomli_w
from string import Template

"""
    Since toml doesn't support environment variables, this method is used in the script 'run.sh' 
    to override the environment variables in the config.template.toml file.
"""

def process_config_template(template_path, output_path):
    """Process the TOML config template and substitute environment variables."""
    with open(template_path, 'r') as f:
        template = Template(f.read())
    
    # Substitute environment variables
    try:
        config_str = template.substitute(os.environ)
    except KeyError as e:
        raise KeyError(f"[-] Missing environment variable: {e}! [-]")
    
    # Parse TOML to validate it
    config_dict = tomli.loads(config_str)
    
    # Write the processed config
    with open(output_path, 'wb') as f:
        tomli_w.dump(config_dict, f)

if __name__ == '__main__':
    template_path = os.getenv("R2R_TEMPLATE_PATH", "./backend/config.template.toml")
    output_path = os.getenv("R2R_CONFIG_PATH", "./backend/config.toml")
    process_config_template(template_path, output_path)