
#!/bin/bash
#set -x

ps_path=/home/wdr/PycharmProjects/wangdingrong/recurrent-visual-attention/hope
png_path=/home/wdr/PycharmProjects/wangdingrong/recurrent-visual-attention/hope_png


cd ${ps_path}
ls -l ${ps_path} |grep "^-" |awk '{print $9}'| while read filein
do
        convert -density 108 -crop 0x0 ${filein} ${png_path}/${filein%%.eps}.png
done
echo "Convert Done"

