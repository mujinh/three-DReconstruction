import zipfile
 
f = zipfile.ZipFile("wtl.zip",'r') # 原压缩文件在服务器的位置
for file in f.namelist():
    f.extract(file,"./")               # 解压到的位置
f.close()
