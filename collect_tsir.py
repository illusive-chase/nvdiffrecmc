text_tsir = '''
Arm RLIT[1] @ PSNR: 29.037
Arm RLIT[1] @ SSIM: 0.9353
Arm RLIT[1] @ LPIPS: 0.0672
Arm RLIT[2] @ PSNR: 28.976
Arm RLIT[2] @ SSIM: 0.9327
Arm RLIT[2] @ LPIPS: 0.0558
Arm RLIT[3] @ PSNR: 29.396
Arm RLIT[3] @ SSIM: 0.9440
Arm RLIT[3] @ LPIPS: 0.0592
Arm RLIT[4] @ PSNR: 29.437
Arm RLIT[4] @ SSIM: 0.9273
Arm RLIT[4] @ LPIPS: 0.0668
Arm RLIT[5] @ PSNR: 30.122
Arm RLIT[5] @ SSIM: 0.9456
Arm RLIT[5] @ LPIPS: 0.0608
Arm Albedo @ PSNR: 31.200
Arm Albedo @ SSIM: 0.9375
Arm Albedo @ LPIPS: 0.1151
Ficus RLIT[1] @ PSNR: 22.471
Ficus RLIT[1] @ SSIM: 0.9275
Ficus RLIT[1] @ LPIPS: 0.0784
Ficus RLIT[2] @ PSNR: 22.732
Ficus RLIT[2] @ SSIM: 0.9226
Ficus RLIT[2] @ LPIPS: 0.0821
Ficus RLIT[3] @ PSNR: 21.462
Ficus RLIT[3] @ SSIM: 0.9271
Ficus RLIT[3] @ LPIPS: 0.0803
Ficus RLIT[4] @ PSNR: 22.741
Ficus RLIT[4] @ SSIM: 0.9288
Ficus RLIT[4] @ LPIPS: 0.0773
Ficus RLIT[5] @ PSNR: 22.157
Ficus RLIT[5] @ SSIM: 0.9339
Ficus RLIT[5] @ LPIPS: 0.0771
Ficus Albedo @ PSNR: 24.914
Ficus Albedo @ SSIM: 0.9077
Ficus Albedo @ LPIPS: 0.0799
Hotdog RLIT[1] @ PSNR: 26.739
Hotdog RLIT[1] @ SSIM: 0.9233
Hotdog RLIT[1] @ LPIPS: 0.1155
Hotdog RLIT[2] @ PSNR: 27.342
Hotdog RLIT[2] @ SSIM: 0.9398
Hotdog RLIT[2] @ LPIPS: 0.1060
Hotdog RLIT[3] @ PSNR: 30.740
Hotdog RLIT[3] @ SSIM: 0.9462
Hotdog RLIT[3] @ LPIPS: 0.0935
Hotdog RLIT[4] @ PSNR: 29.344
Hotdog RLIT[4] @ SSIM: 0.9314
Hotdog RLIT[4] @ LPIPS: 0.1114
Hotdog RLIT[5] @ PSNR: 31.627
Hotdog RLIT[5] @ SSIM: 0.9519
Hotdog RLIT[5] @ LPIPS: 0.0928
Hotdog Albedo @ PSNR: 30.628
Hotdog Albedo @ SSIM: 0.9678
Hotdog Albedo @ LPIPS: 0.0521
Lego RLIT[1] @ PSNR: 23.907
Lego RLIT[1] @ SSIM: 0.8637
Lego RLIT[1] @ LPIPS: 0.1165
Lego RLIT[2] @ PSNR: 21.921
Lego RLIT[2] @ SSIM: 0.8673
Lego RLIT[2] @ LPIPS: 0.1122
Lego RLIT[3] @ PSNR: 26.832
Lego RLIT[3] @ SSIM: 0.9062
Lego RLIT[3] @ LPIPS: 0.0938
Lego RLIT[4] @ PSNR: 25.965
Lego RLIT[4] @ SSIM: 0.8947
Lego RLIT[4] @ LPIPS: 0.0936
Lego RLIT[5] @ PSNR: 27.298
Lego RLIT[5] @ SSIM: 0.9109
Lego RLIT[5] @ LPIPS: 0.0893
Lego Albedo @ PSNR: 24.100
Lego Albedo @ SSIM: 0.9108
Lego Albedo @ LPIPS: 0.0842
'''

if __name__ == '__main__':

    def avg(l, fixed):
        fmt = f'.{fixed}f'
        return f'{sum(l) / len(l):{fmt}}'

    def display(l):
        print(' & '.join([str(x) for x in l]))

    lut = {}
    for key, value in [line.split(': ') for line in text_tsir.split('\n') if line]:
        lut.setdefault(key.split(' ', 1)[0], {})[key.split(' ', 1)[1]] = float(value)

    scenes = ['Lego', 'Hotdog', 'Arm', 'Ficus']

    print('Relight')
    lst = []
    for scene in scenes:
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ PSNR'] for i in range(5)]) / 5, 3))
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ SSIM'] for i in range(5)]) / 5, 4))
        lst.append(round(sum([lut[scene][f'RLIT[{i+1}] @ LPIPS'] for i in range(5)]) / 5, 4))
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)

    print('Albedo')
    lst = []
    for scene in scenes:
        lst.append(lut[scene]['Albedo @ PSNR'])
        lst.append(lut[scene]['Albedo @ SSIM'])
        lst.append(lut[scene]['Albedo @ LPIPS'])
    lst.append(avg(lst[0::3], 2))
    lst.append(avg(lst[1::3], 4))
    lst.append(avg(lst[2::3], 4))
    display(lst)