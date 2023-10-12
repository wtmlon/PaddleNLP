import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer 
#tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", )
#model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", dtype='float16', convert_from_torch=True)
tokenizer = AutoTokenizer.from_pretrained("qwen/qwen-14b", )
model = AutoModelForCausalLM.from_pretrained("qwen/qwen-14b", dtype='bfloat16')
#inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pd')
input_ids = paddle.to_tensor([[111,223,323]])
print(model(input_ids)[0])
#pred = model.generate(**inputs, max_length=64, decode_strategy='greedy_search')[0]
#print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
import pdb; pdb.set_trace()
