from datasets import load_dataset
import pandas as pd
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

raw_datasets = load_dataset("code_search_net", "python", split='test', trust_remote_code=True)
raw_datasets = raw_datasets.select(range(1000))
raw_datasets = pd.DataFrame(raw_datasets)
raw_datasets.to_excel("./raw.xlsx")
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def delete_syms(delete, code):
    res = ""
    i, j = delete[0]
    res += code[0:i]
    for k, p in delete[1:len(delete):]:
        i = k
        res += code[j:i]
        j = p
    res += code[j:]
    return res

def remove_comments(node):
    res = list()
    for i in node.children:
        if (i.type == 'comment'):
            res.append((i.start_byte, i.end_byte)) 
        if (i.type == 'expression_statement' and i.children[0].type == 'string' and i.parent.parent.type == 'function_definition'):
            res.append((i.start_byte, i.end_byte)) 
        res += remove_comments(i)
    return res

def parse_code(code):
    tree = parser.parse(bytes(code, "utf8"))
    body = code.split(":", 1)[1]
    function_name = None
    

    for child in tree.root_node.children:
        if child.type == 'function_definition':
            function_name = child.child_by_field_name('name').text.decode('utf-8')
            body_with_comments = child.text.decode('utf-8')
            to_delete = remove_comments(child.child_by_field_name('body'))
            if (to_delete):
                body_without_comments = delete_syms(to_delete, code)
            else:
                body_without_comments =  child.text.decode('utf-8')
            body_without_comments = body_without_comments.split('\n')
            body_without_comments = [line for line in body_without_comments if line.strip() != '']
            body_without_comments = '\n'.join(body_without_comments)
            body_without_comments = body_without_comments.replace(function_name, "<extra_id_0>", 1)
            body_with_comments = body_with_comments.replace(function_name, "<extra_id_0>", 1)

    return (function_name, body_with_comments, body_without_comments)


# code_example = """
# def greet(name):
#     \"\"\"This function greets a person.\"\"\"
#     print("Hello, " + name + "!")  # Greeting message
# """

# code_example2="""
# def sina_xml_to_url_list(xml_data):
#     \"\"\"str->list
#     Convert XML to URL List.
#     From Biligrab.
#     \"\"\"
#     rawurl = []
#     # Comment1
#     # Comment 2
#     dom = parseString(xml_data)
#     for node in dom.getElementsByTagName('durl'):
#         url = node.getElementsByTagName('url')[0]  # Comment 3
#         rawurl.append(url.childNodes[0].data)
#     return rawurl
# """

# first = parse_code(code_example)
# second = parse_code(code_example2)

# print(first)
# print(second)

func_names = list()
bwc = list()
bnc = list()
data = raw_datasets['whole_func_string']
wrong = 0
for i in data:
    res = parse_code(i)
    func_names.append(res[0])
    bwc.append(res[1])
    bnc.append(res[2])

raw_datasets['my_func_name'] = func_names
raw_datasets['my_bwc'] = bwc
raw_datasets['my_bnc'] = bnc
raw_datasets.to_excel("./prepared.xlsx")

for i in range(1000):
    tmp = raw_datasets['func_name'][i].rsplit(".", 1)
    if len(tmp) > 1:
        tmp = tmp[1]
    else:
        tmp = tmp[0]
    tmp2 = raw_datasets['my_func_name'][i]
    print(tmp)
    if tmp2 != tmp:
        wrong += 1

print(raw_datasets['whole_func_string'].iloc[15])
print("="*10)
print(raw_datasets['my_func_name'].iloc[15])
print("="*10)
print(raw_datasets['my_bwc'].iloc[15])
print("="*10)
print(raw_datasets['my_bnc'].iloc[15])
print("="*10)
print("Number of wrong func names:", wrong)
