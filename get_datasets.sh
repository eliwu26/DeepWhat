cd /
mkdir datasets
cd datasets

wget https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.train.jsonl.gz https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.valid.jsonl.gz https://s3-us-west-2.amazonaws.com/guess-what/guesswhat.test.jsonl.gz
gunzip *.gz
rm *.gz

mkdir coco
cd coco
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip http://msvocds.blob.core.windows.net/coco2014/val2014.zip

unzip train2014.zip
cd train2014
echo *.jpg | xargs mv -t ..
cd ..
rm -rf train2014/
rm train2014.zip

unzip val2014.zip
cd val2014
echo *.jpg | xargs mv -t ..
cd ..
rm -rf val2014/
rm val2014.zip
