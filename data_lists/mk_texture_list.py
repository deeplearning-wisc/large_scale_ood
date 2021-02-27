import os

root_dir = "/nobackup-slow/dataset/dtd/images"

sub_dirs = os.listdir(root_dir)

lines = []
for i, sub_dir in enumerate(sorted(sub_dirs)):
    all_imgs = os.listdir(os.path.join(root_dir, sub_dir))
    for img in sorted(all_imgs):
        lines.append("{} {}".format('images/'+sub_dir+'/'+img, i))

with open('textures_selected_list.txt', 'w') as f:
    for line in lines:
        f.write(line + '\n')
