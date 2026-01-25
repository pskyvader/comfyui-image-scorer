import sys, importlib.util, os
p = r'D:/Programming/Python/comfyui-image-scorer/full_data'
print('adding to sys.path:', p)
sys.path.insert(0, p)
print('sys.path[0]=', sys.path[0])
print('find_spec(training)=', importlib.util.find_spec('training'))
print('listdir full_data=', os.listdir(p))
