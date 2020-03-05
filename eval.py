import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer

sentences = [
    '안녕하세요 여러분 저는 음성합성 모델입니다.',
    '밥을 많이 먹다보면 배가 정말 많이 부릅니다.저는 밥을 많이 먹지 않겠습니다.',
    '이건, 바다쥐만의 문제가 아니야! 문명화된 인간은, 위대한 자연의 법칙에서 벗어나, 점점 더 용서받을 수 없는 큰 죄를 범하고 있다!',
    '영화애호가들이라면 메리크리스마스 미스터 로렌스라는 그의 83년도 작품이, 국제적으로 높은 평가를 받은 수작이라는 것.. 다 아실겁니다.',
    '나.. 기억이 돌아왔어. 그렇지만 달라진 건 아무것도 없어. 여기밖에 돌아올 데가 없었다고! 그런데.. 어딜 가는 거야.. 뭐하러 가냐고!'
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    main()
