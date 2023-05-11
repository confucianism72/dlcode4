import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class BaseModel(nn.Module):

    def __init__(self, args, dictionary):
        super().__init__()
        self.padding_idx = dictionary.pad()
        self.dictionary = dictionary


class LMModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        # Hint: Use len(dictionary) in __init__
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################

    def logits(self, source, **unused):
        """
        Compute the logits for the given source.

        Args:
            source: The input data.
            **unused: Additional unused arguments.

        Returns:
            logits: The computed logits.
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        return logits

    def get_loss(self, source, target, reduce=True, **unused):
        logits = self.logits(source)
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1))
        return F.nll_loss(
            lprobs,
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, prefix, max_len=100, beam_size=None):
        """
        Generate text using the trained language model with beam search.

        Args:
            prefix (str): The initial words, like "白".
            max_len (int, optional): The maximum length of the generated text.
                                     Defaults to 100.
            beam_size (int, optional): The beam size for beam search. Defaults to None.

        Returns:
            outputs (str): The generated text.(e.g. "白日依山尽，黄河入海流，欲穷千里目，更上一层楼。")
        """
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        raise NotImplementedError()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        outputs = ""
        return outputs


#'ids'
#'lengths'
#'source'
#'prev_outputs'
#'target'


# "--embedding-dim", 
# "--hidden-size", 
# "--num-layers", 

class Seq2SeqModel(BaseModel):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary, )
        # Hint: Use len(dictionary) in __init__
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        #ENC_HID_DIM = args.hidden_size
        self.device = torch.device('mps') if torch.backends.mps.is_available()  else  (torch.device(
                                    "cuda") if torch.cuda.is_available() else torch.device("cpu"))

        attn = Attention(args.hidden_size, args.hidden_size)
        INPUT_DIM = len(dictionary)
        OUTPUT_DIM = len(dictionary)
        ENC_DROPOUT = args.dropout_input
        DEC_DROPOUT = args.dropout_output
        encoder = Encoder(INPUT_DIM, args.embedding_dim , args.hidden_size, args.hidden_size, ENC_DROPOUT,num_layers= args.num_layers).to(self.device)
        decoder = Decoder(OUTPUT_DIM, args.embedding_dim , args.hidden_size, args.hidden_size, DEC_DROPOUT, attn).to(self.device)
        
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = args.teacher_forcing_ratio
        #self.src_pad_idx = padding_idx
        #self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index = self.padding_idx)
        
        self.init_weights()
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################O
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def create_mask(self, src):
        mask = (src != self.padding_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
                    
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        mask = self.create_mask(src)

        #mask = [batch size, src len]
                
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        # print('forward',outputs.shape,outputs.device)
        return outputs

    def logits(self, source, prev_outputs,lengths,eval = 1, **unused):
        """
        Compute the logits for the given source and previous outputs.

        Args:
            source: The input data.
            prev_outputs: The previous outputs.
            **unused: Additional unused arguments.

        Returns:
            logits: The computed logits.
        """
        source = source.permute(1,0)
        prev_outputs = prev_outputs.permute(1,0)
        tf = self.teacher_forcing_ratio if eval == 0 else 0
        output = self.forward(source, lengths, prev_outputs,teacher_forcing_ratio= tf)  #(src, src_len, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        #output = output[1:].view(-1, output_dim)

        logits = output#[1:]

        #trg = prev_outputs[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        # logits = self.criterion(output, trg)
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        # print('logits',logits.shape,logits.device)
        return logits.permute(1,0,2)

    def get_loss(self, source, prev_outputs, target, lengths, eval, reduce=True,  **unused):
        # print('get_loss',source.shape, prev_outputs.shape, target.shape, lengths.shape)
        logits = self.logits( source, prev_outputs, lengths,eval = eval , **unused) # ource, prev_outputs,lengths,
        lprobs = F.log_softmax(logits, dim=-1).view(-1, logits.size(-1)) 
        # print(logits.shape, lprobs.shape, target.shape)
        # print(logits.device, lprobs.device, target.device)#, self.padding_idx.device)
        return F.nll_loss(
            lprobs,
            target.view(-1),
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )

    @torch.no_grad()
    def generate(self, inputs, max_len=100, beam_size=None):
        """
        Generate text using the trained sequence-to-sequence model with beam search.

        Args:
            inputs (str): The input text, e.g., "改革春风吹满地".
            max_len (int, optional): The maximum length of the generated text.
                                     Defaults to 100.
            beam_size (int, optional): The beam size for beam search. Defaults to None.

        Returns:
            outputs (str): The generated text, e.g., "复兴政策暖万家".
        """
        # Hint: Use dictionary.encode_line and dictionary.bos() or dictionary.eos()
        ##############################################################################
        #                  TODO: You need to complete the code here                  #
        ##############################################################################
        in_token = inputs.strip().replace(" ", "")
        in_token =[self.dictionary.bos()] + self.dictionary.encode_line(inputs)  + [self.dictionary.eos()] # ,add_if_not_exist=False,append_eos=False).long()
        src_len = torch.LongTensor([len(in_token)])
        src_tensor = torch.LongTensor(in_token).unsqueeze(1)
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src_tensor, src_len)

        mask = self.create_mask(src_tensor)
            
        trg_indexes = [[self.dictionary.bos(),1]]

        # attentions = torch.zeros(max_len, 1, len(in_token))
        
        end = self.dictionary.eos()

        if beam_size is None:
            beam_size = 1

        for i in range(max_len):
            new = []
            t_list = []
            for trg_tokens in trg_indexes:
                trg_tensor = torch.LongTensor([trg_tokens[0][-1]])                   
                with torch.no_grad():
                    output, hidden, attention = self.decoder(trg_tensor, hidden, encoder_outputs, mask)
                # attentions[i] = attention
                # output = [batch size, output dim]

                output = F.log_softmax(output, dim=1)
                prob , topk = output.topk(beam_size, dim=1)
                # pred_token = output.argmax(1).item()
                new1 = [[trg_tokens[0] + [t], trg_tokens[1] + prob[t] if t != end else  trg_tokens[1] ] for t in topk]
                # trg_tokens.append(pred_token)
                new.extend(new1)
                t_list.extend(topk)
            new.sort(key=lambda x: x[1], reverse=True)
            trg_indexes = new[0:beam_size]

            if all(t == end for t in t_list): #trg_field.vocab.stoi[trg_field.eos_token]:
                break
        #trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        
        outputs = self.dictionary.decode_line(trg_indexes[1:])
        return outputs #trg_tokens[1:]  #, attentions[:len(trg_tokens)-1]
        
        ##############################################################################
        #                              END OF YOUR CODE                              #
        ##############################################################################
        #outputs = ""
        #return outputs

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True,num_layers = num_layers)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
                
        #need to explicitly put lengths on cpu!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu')) #####
                
        packed_outputs, hidden = self.rnn(packed_embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
  
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention = [batch size, src len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim = 1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs, mask)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden.squeeze(0), a.squeeze(1)
