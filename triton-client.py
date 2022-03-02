"""Client for connecting with Triton TTS server."""
import torch
import grpc
import numpy as np

from scipy.io.wavfile import write


try:
    from tritonclient.grpc import (
        InferInput, InferenceServerClient, InferenceServerException)
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'install Triton client library by running "pip install nvidia-pyindex",'
        ' followed by "pip install tritonclient[grpc]==2.8.0"'
    )


class TtsClient:
    """Client for connecting with Triton TTS server.

    The inference functions are hardcoded for the following model
    configuration on Triton:

    input {
        name: "input__0"
        data_type: TYPE_INT64
        dims: [-1]
      }
    output [
      {
        name: "output__0"
        data_type: TYPE_INT16
        dims: [-1]
      }
    ]
    dynamic_batching {}
    instance_group {
          count: 1
      }
    """

    @staticmethod
    def create(host='localhost', port=8001, model_name='nemo_model1'):
        """Static factory method.

        The destructor is not called properly if InferenceServerClient is
        constructed in __main__.

        Args:
            host: server IP address
            port: server port number
            model_name: name of model on Triton server to use

        Returns:
            TtsClient.

        Raises:
            ConnectionError: failed to connect to server
        """
        try:
            #client = grpc.insecure_channel(f'{host}:{port}')
            client = InferenceServerClient(f'{host}:{port}')
            client.is_server_ready()
        except InferenceServerException:
            raise ConnectionError(
                f'Failed to connect to Triton server at [{host}:{port}]')
        return TtsClient(client, model_name)

    def __init__(self, client, model_name):
        self._client = client
        self._model_name = model_name

    def infer(self, text: str) -> np.ndarray:
        """Base function for inference with Triton server.

        Args:
            audio: signed int64 tensor of shape [tokens]

        Returns:
            Decoded string.
        """
        tokens = np.array([[str(c).encode('utf-8') for c in text]],
                          dtype=np.object_)
        input0 = InferInput('input__0', tokens.shape, 'BYTES')
        input0.set_data_from_numpy(tokens)
        output = self._client.infer(self._model_name, inputs=[input0])
        return output.as_numpy('output__0')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('text', type=str, help='text for speech synthesis')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='server IP address')
    parser.add_argument('--port', type=int, default='8001',
                        help='server port number')
    parser.add_argument('--model', type=str, default='nemo_model1',
                        help='name of model on Triton server to use')
    parser.add_argument('-o', '--out_file', type=str,
                            help='path to write WAV file')
    args = parser.parse_args()
    client = TtsClient.create(args.host, args.port, args.model)
    audio = client.infer(args.text)
    if args.out_file:
        write(args.out_file, 22050, audio)
    else:
        print(len(audio), audio)
