import os
import torch
from builder.Layer import EncoderRNN, GreedySearchDecoder, LuongAttnDecoderRNN
from utils.Config import MAX_LENGTH, MIN_COUNT
from utils.Data import loadConversations, loadLines
from utils.Preprocessing import Voc, loadPrepareData, trimRareWords
from builder.Config import hidden_size,encoder_n_layers,decoder_n_layers,checkpoint_iter,device,dropout,attn_model
from utils.Preprocessing import indexesFromSentence, normalizeString
import torch.nn as nn
from torch import optim

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("data", corpus_name)

# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")


# Load/Assemble voc and pairs
save_dir = os.path.join("runs", "train")
voc = Voc(corpus_name)


model_name = 'cb_model'
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                           '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                           '{}_checkpoint.tar'.format(checkpoint_iter))

if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()
print('Building encoder and decoder is done!')
# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

print('Start chatting!')
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

# Begin chatting (uncomment and run the following line to begin)
def ChatWithBot(input_text):
    input_sentence = normalizeString(input_text)
    output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
    return ' '.join(output_words)