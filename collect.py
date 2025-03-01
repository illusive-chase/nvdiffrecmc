text_s4r='''
Air RLIT[1] @ PSNR: 28.618
Air RLIT[1] @ SSIM: 0.9251
Air RLIT[1] @ LPIPS: 0.1040
Air RLIT[2] @ PSNR: 28.908
Air RLIT[2] @ SSIM: 0.9386
Air RLIT[2] @ LPIPS: 0.0990
Air Albedo @ PSNR: 25.225
Air Albedo @ SSIM: 0.9209
Air Albedo @ LPIPS: 0.0580
Air Roughness @ MSE: 0.003
Chair RLIT[1] @ PSNR: 31.496
Chair RLIT[1] @ SSIM: 0.9615
Chair RLIT[1] @ LPIPS: 0.0525
Chair RLIT[2] @ PSNR: 31.823
Chair RLIT[2] @ SSIM: 0.9614
Chair RLIT[2] @ LPIPS: 0.0550
Chair Albedo @ PSNR: 30.396
Chair Albedo @ SSIM: 0.9418
Chair Albedo @ LPIPS: 0.0647
Chair Roughness @ MSE: 0.005
Jugs RLIT[1] @ PSNR: 29.536
Jugs RLIT[1] @ SSIM: 0.9436
Jugs RLIT[1] @ LPIPS: 0.0712
Jugs RLIT[2] @ PSNR: 30.269
Jugs RLIT[2] @ SSIM: 0.9475
Jugs RLIT[2] @ LPIPS: 0.0647
Jugs Albedo @ PSNR: 31.304
Jugs Albedo @ SSIM: 0.9472
Jugs Albedo @ LPIPS: 0.0817
Jugs Roughness @ MSE: 0.021
Hotdog RLIT[1] @ PSNR: 30.916
Hotdog RLIT[1] @ SSIM: 0.9489
Hotdog RLIT[1] @ LPIPS: 0.0959
Hotdog RLIT[2] @ PSNR: 30.261
Hotdog RLIT[2] @ SSIM: 0.9444
Hotdog RLIT[2] @ LPIPS: 0.0982
Hotdog Albedo @ PSNR: 29.644
Hotdog Albedo @ SSIM: 0.9699
Hotdog Albedo @ LPIPS: 0.0630
Hotdog Roughness @ MSE: 0.010
'''


if __name__ == '__main__':

    def avg(l, fixed):
        fmt = f'.{fixed}f'
        return f'{sum(l) / len(l):{fmt}}'

    def display(l):
        print(' & '.join([str(x) for x in l]))

    lut = {}
    for key, value in [line.split(': ') for line in text_s4r.split('\n') if line]:
        lut.setdefault(key.split(' ', 1)[0], {})[key.split(' ', 1)[1]] = float(value)

    scenes = ['Air', 'Chair', 'Hotdog', 'Jugs']
    # scenes = ['Lego', 'Hotdog', 'Armadillo', 'Ficus']

    # print('NVS')
    # lst = []
    # for scene in scenes:
    #     lst.append(lut[scene]['NVS @ PSNR'])
    #     lst.append(lut[scene]['NVS @ SSIM'])
    #     lst.append(lut[scene]['NVS @ LPIPS'])
    # lst.append(avg(lst[0::3], 2))
    # lst.append(avg(lst[1::3], 4))
    # lst.append(avg(lst[2::3], 4))
    # display(lst)

    # print('Envmap6')
    # lst = []
    # for scene in scenes:
    #     lst.append(lut[scene]['RLIT[1] @ PSNR'])
    #     lst.append(lut[scene]['RLIT[1] @ SSIM'])
    #     lst.append(lut[scene]['RLIT[1] @ LPIPS'])
    # lst.append(avg(lst[0::3], 2))
    # lst.append(avg(lst[1::3], 4))
    # lst.append(avg(lst[2::3], 4))
    # display(lst)

    # print('Envmap12')
    # lst = []
    # for scene in scenes:
    #     lst.append(lut[scene]['RLIT[2] @ PSNR'])
    #     lst.append(lut[scene]['RLIT[2] @ SSIM'])
    #     lst.append(lut[scene]['RLIT[2] @ LPIPS'])
    # lst.append(avg(lst[0::3], 2))
    # lst.append(avg(lst[1::3], 4))
    # lst.append(avg(lst[2::3], 4))
    # display(lst)

    print('Relight')
    lst = []
    for scene in scenes:
        lst.append(round((lut[scene]['RLIT[1] @ PSNR'] + lut[scene]['RLIT[2] @ PSNR']) / 2, 3))
        lst.append(round((lut[scene]['RLIT[1] @ SSIM'] + lut[scene]['RLIT[2] @ SSIM']) / 2, 4))
        lst.append(round((lut[scene]['RLIT[1] @ LPIPS'] + lut[scene]['RLIT[2] @ LPIPS']) / 2, 4))
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

    print('MSE')
    lst = []
    for scene in scenes:
        lst.append(lut[scene]['Roughness @ MSE'])
    lst.append(avg(lst, 3))
    display(lst)