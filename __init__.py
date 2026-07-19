import os
import importlib.util
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

MSGCP = "ComfyUI-CustomPack"
flag = True

class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"
    
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

def load_nodes_recursive(nodes_dir):
    if not os.path.exists(nodes_dir):
        return
    
    for root, dirs, files in os.walk(nodes_dir):
        dirs[:] = [d for d in dirs if not d.startswith('_') and d != '__pycache__']
        
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                try:
                    spec = importlib.util.spec_from_file_location(f"nodes_{file[:-3]}", file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for attr_name in dir(module):
                        if attr_name.startswith('NODE_CLASS_MAPPINGS_'):
                            class_mappings = getattr(module, attr_name)
                            
                            display_attr_name = attr_name.replace('CLASS', 'DISPLAY_NAME')
                            
                            if hasattr(module, display_attr_name):
                                display_mappings = getattr(module, display_attr_name)
                            else:
                                display_mappings = {}
                                for key in class_mappings.keys():
                                    display_mappings[key] = key
                            
                            NODE_CLASS_MAPPINGS.update(class_mappings)
                            NODE_DISPLAY_NAME_MAPPINGS.update(display_mappings)
                            
                            rel_path = os.path.relpath(file_path, nodes_dir)
                            print(f"{MSGCP}: {Colors.GREEN}Loaded {len(class_mappings)} nodes from {rel_path}{Colors.RESET}", file=sys.__stderr__)
                            
                except Exception as e:
                    flag = False
                    rel_path = os.path.relpath(file_path, nodes_dir)
                    print(f"{MSGCP}: {Colors.RED} ERROR in {rel_path}: {e} {Colors.RESET}", file=sys.__stderr__)

package_dir = os.path.dirname(__file__)
nodes_dir = os.path.join(package_dir, "nodes")

if os.path.exists(nodes_dir):
    load_nodes_recursive(nodes_dir)
    print(f"{MSGCP}: Total nodes = {len(NODE_CLASS_MAPPINGS)}")
else:
    print(f"Folder 'nodes' not found in {package_dir}")

if flag == True:
    print(f"{MSGCP}: {Colors.GREEN}Success...{Colors.RESET}", file=sys.__stderr__)
else:
    print(f"{MSGCP}: {Colors.RED}Loaded with errors!{Colors.RESET}", file=sys.__stderr__)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

