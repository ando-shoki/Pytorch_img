def gamma(img, gamma=1.8):
    """RGB画像にガンマ補正をかける。

    R/G/B で同じ LUT を使うため、LUT は1回だけ作って3チャンネル分つなげる。
    PIL の ``Image.point`` は RGB の場合 256 * 3 要素の LUT を受け取る。
    """
    lut_values = [int(255 * (float(i) / 255) ** (1.0 / gamma)) for i in range(256)]
    lut = lut_values + lut_values + lut_values

    return img.point(lut)
