# JDAKD
This is the source code of JDAKD model.

## Requirements
* Python (tested on 3.7.4)
* CUDA (tested on 10.2)
* [PyTorch](http://pytorch.org/) (tested on 1.7.1)
* [Transformers](https://github.com/huggingface/transformers) (tested on 2.11.0)

## Pretrained models
Download pretrained language model from [huggingface](https://huggingface.co/bert-base-uncased) and put it into the `./bert-base-uncased` directory. 

## Run
### FewRel
Train the JDAKD model on FewRel dataset under 10-way-5-shot (10-way-10-shot) setting with the following command:

```bash
>> python main.py --task FewRel --shot 5  # for 10-way-5-shot setting
>> python main.py --task FewRel --shot 10 # for 10-way-10-shot setting 
```

### TACRED
Train the JDAKD model on TACRED dataset under 5-way-5-shot (5-way-10-shot) setting with the following command:
```bash
>> python main.py --task tacred --shot 5  # for 5-way-5-shot setting
>> python main.py --task tacred --shot 10  # for 5-way-10-shot setting
```


## Citation

If you find the repository helpful, please cite the following paper.
```

```