from . import LlamaPretrainedModel
from paddlenlp.transformers.model_outputs import (
    CausalLMOutputWithCrossAttentions,
)
import paddle.nn as nn
import paddle

#class SiblingLlama(LlamaPretrainedModel):
class SiblingLlama(nn.Layer):
    base_model_prefix = "llama"
    def __init__(self, model1, model2):
        super().__init__()
        self.model1, self.model2 = model1, model2
        self.vocab_size = self.model1.config.vocab_size
        self.down1 = nn.Linear(self.vocab_size, self.vocab_size // 20)
        self.up1 = nn.Linear(self.vocab_size // 20, self.vocab_size)
        self.down2 = nn.Linear(self.vocab_size, self.vocab_size // 20)
        self.up2 = nn.Linear(self.vocab_size // 20, self.vocab_size)
        #self.loss_fn = nn.KLDivLoss()
        self.loss_fn = nn.MSELoss()
        self.scale = 0.1
    
    def forward(
        self,
        *args,
        **kwargs,
    ):
        output1 = self.model1(*args, **kwargs)
        output2 = self.model2(*args, **kwargs)

        loss_cross = 0.0
        if output1[0] is not None and output2[0] is not None:
            ffn_out1 = self.up1(self.down1(output1[1]))
            ffn_out2 = self.up2(self.down2(output2[1]))
            loss_cross = self.loss_fn(ffn_out1, ffn_out2)
        
        #print("loss 1:", output1[0], "loss 2:", output2[0], "cross loss: ", loss_cross)
        loss = output1[0] + output2[0] + loss_cross * self.scale

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            #logits=paddle.concat([output1[1], output2[1]], axis=0),
            logits=output1[1],
        )

