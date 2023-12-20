from . import LlamaPretrainedModel
from paddlenlp.transformers.model_outputs import (
    CausalLMOutputWithCrossAttentions,
)
import paddle.nn as nn
import paddle.nn.functional as F
import paddle

#class SiblingLlama(LlamaPretrainedModel):
class SiblingLlama(nn.Layer):
    base_model_prefix = "llama"
    def __init__(self, model1, model2):
        super().__init__()
        self.model1, self.model2 = model1, model2
        self.hidden_size = self.model1.config.hidden_size
        #self.down1 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.up1 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.relu = nn.ReLU()
        #self.down2 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.up2 = nn.Linear(self.hidden_size, self.hidden_size)
        #self.loss_fn = nn.KLDivLoss()
        self.loss_fn = nn.MSELoss()
        self.scale = 1
    
    def forward(
        self,
        *args,
        **kwargs,
    ):
        #kwargs["output_hidden_states"] = True
        output1 = self.model1(*args, **kwargs)
        output2 = self.model2(*args, **kwargs)

        loss_cross = 0.0
        if output1[0] is not None and output2[0] is not None:
            #ffn_out1 = self.up1(self.relu(self.down1(output1[2][-1])))
            #ffn_out2 = self.up2(self.relu(self.down2(output2[2][-1])))
            #loss_cross = self.loss_fn(ffn_out1, ffn_out2)
            student = F.log_softmax(output1[1].astype("float32"), axis=2)
            teacher = F.softmax(output2[1].astype("float32"), axis=2)

            loss_cross = F.kl_div(student, teacher, reduction="batchmean")
            #loss_cross = paddle.clip(loss_cross, max=15)
            #loss_cross = self.loss_fn(output1[1], output2[1])
        
        print("loss 1:", output1[0], "loss 2:", output2[0], "cross loss: ", loss_cross)
        loss = output1[0] + output2[0] + loss_cross * self.scale

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            #logits=paddle.concat([output1[1], output2[1]], axis=0),
            logits=output2[1],
        )

