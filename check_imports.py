try:
    from langchain.output_parsers import StructuredOutputParser
    print('Found in langchain')
except ImportError as e:
    print(f'Not in langchain: {e}')

try:
    from langchain_core.output_parsers import StructuredOutputParser
    print('Found in core')
except ImportError as e:
    print(f'Not in core: {e}')
