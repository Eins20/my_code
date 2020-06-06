import os
import shutil

# files = sorted(os.listdir("/home/ices/work/tzh/Unet/raw_images"))
# for i in range(int(len(files)/20)):
#     if not os.path.exists(os.path.join("/extend/my_data/train",str(i+100))):
#         os.makedirs(os.path.join("/extend/my_data/train",str(i+100)))
#     for file in files[i*20:(i+1)*20]:
#         file_name = file.split('.')[0]+".ref"
#         year = file_name[12:14]
#         src_path = os.path.join("/extend/14-17_2500_radar/",year+"_2500_radar",file_name)
#         des_path = os.path.join("/extend/my_data/train",str(i+100),file_name)
#         shutil.copyfile(src_path,des_path)

files = sorted(os.listdir("/extend/14-17_2500_radar/17_2500_radar"))
# print(files)
i = files.index("cappi_ref_201710151300_2500_0.ref")
for file in files[i:i+100]:
    shutil.copyfile(os.path.join("/extend/14-17_2500_radar/17_2500_radar",file),
                    os.path.join("/extend/my_data_test/test_data",file))
