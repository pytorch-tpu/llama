import time
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np
from llama.tokenizer import Tokenizer

GRPC_PORT = "8500"
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3  # Max LENGTH the GRPC should handle
tokenizer_path = 'tokenizer.model'

def serve_grpc(tokenizer, sentence):
    channel = grpc.insecure_channel(f'localhost:{GRPC_PORT}',
                                    options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = 'tfllama'
    grpc_request.model_spec.signature_name = 'serving_default'


    tokenized = tokenizer.encode(sentence, bos=True, eos=False)

    if len(tokenized) < 32:
        tokenized = [tokenizer.pad_id] * (32 - len(tokenized)) + tokenized

    tokens = tf.cast(tf.constant(tokenized), tf.int64)
    grpc_request.inputs['input_tokens'].CopyFrom(tf.make_tensor_proto(tokens))

    start = time.time()
    predictions = stub.Predict(grpc_request)
    end = time.time()

    # converting from tensor proto to numpy
    res = []
    real_length = 0
    for i, t in enumerate(predictions.outputs['output_0'].int64_val):
        if t == tokenizer.eos_id:
            break
        res.append(t)
    print(tokenizer.decode(res))
    print('Returned 96 tokens, with {} tokens before EOS', len(res))
    print('RPC time: {}, tokens per second {}'.format(end - start, 95 / (end - start)))


def main():
    tokenizer = Tokenizer(model_path=tokenizer_path)
    while True:
        input_sentence = input('Type a sentence to be completed: ')
        serve_grpc(tokenizer, input_sentence)


if __name__ == '__main__':
    main()
