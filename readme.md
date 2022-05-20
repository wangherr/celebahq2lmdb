## A script to transform celebaHQ dataset to lmdb format

-----------------------
original celebahq directory level
~~~
|-celeba-tfr
    |-train
        |-train-r08-s-0000-of-0120.tfrecords
        ...
    |-validation
        |-validation-r08-s-0000-of-0040.tfrecords
        ...
~~~

lmdb directory level
~~~
|-celeba_lmdb
    |-train.lmdb
        |-data.mdb
        |-lock.mdb
    |-validation.lmdb
        |-data.mdb
        |-lock.mdb
~~~


## use

1. download and unzip celebahq (cite: https://github.com/openai/glow)
~~~
wget https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar
tar -xvf celeb-tfr.tar
~~~
2. run the follow command (rewrite from: https://github.com/NVlabs/NVAE)
~~~
python create_celebaAttr_lmdb.py --tfr_path=$DATA_DIR/celeba/celeba-tfr --lmdb_path=$DATA_DIR/celeba/celeba-lmdb --split=train
python create_celebaAttr_lmdb.py --tfr_path=$DATA_DIR/celeba/celeba-tfr --lmdb_path=$DATA_DIR/celeba/celeba-lmdb --split=validation
~~~
3. use in dataset
~~~
    def __getitem__(self, index):
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            img = np.asarray(data, dtype=np.uint8)
            size = int(np.sqrt(len(img) / 3))
            img = np.reshape(img, (size, size, 3))
            img = Image.fromarray(img, mode='RGB')

            attr = txn.get(('a'+str(index)).encode())
            attr_array = np.asarray(attr, dtype=np.int8)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
~~~
