from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-7B", )
#model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-7B", dtype='float16', convert_from_torch=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')
input_ids = torch.tensor([[111,223,323]]).to('cuda:0')
print(model(input_ids)[0])
#inputs = tokenizer('蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是', return_tensors='pt')
#inputs = inputs.to('cuda:0')
#pred = model.generate(**inputs, do_sample=False, num_beams=1, max_length=64)
#print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
import pdb; pdb.set_trace()
