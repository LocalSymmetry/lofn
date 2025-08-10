import base64
import ast
from pathlib import Path

# Load only the prepare_image_messages function without importing heavy dependencies
source = Path('lofn/llm_integration.py').read_text()
module = ast.parse(source)
func_node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare_image_messages')
module_ast = ast.Module(body=[func_node], type_ignores=[])
code = compile(module_ast, filename="<prepare_image_messages>", mode="exec")

class HumanMessage:
    def __init__(self, content):
        self.content = content

ns = {'HumanMessage': HumanMessage, 'List': list}
exec(code, ns)
prepare_image_messages = ns['prepare_image_messages']


def test_prepare_image_messages_limit():
    dummy = base64.b64encode(b'test').decode()
    images = [f"data:image/png;base64,{dummy}" for _ in range(6)]
    msgs = prepare_image_messages(images)
    assert len(msgs) == 5
    for m in msgs:
        assert m.content[0]["type"] == "image_url"
