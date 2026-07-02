"""
계측 레시피 템플릿 — IPDK 배포용 골격
=====================================

이 파일은 '배포 가능한 코드'의 표준 형태를 보여주는 골격입니다.
- 측정 전체를 measure() 함수 하나로 묶었습니다 (셀 순서 의존 없음).
- 파일 경로를 코드에 박지 않고, 이미지·scale·파라미터를 인자로 받습니다.
- 설정은 PARAMS 로 분리했습니다.
- 실패해도 멈추지 않고 status 를 담은 dict 를 반환합니다.
- 같은 입력에는 항상 같은 출력이 나옵니다(무작위 요소 없음).

※ IPDK 고유의 RCP 패키징·입출력 후크·메타데이터 형식은 사내 IPDK 문서에 맞춰
   맨 아래 '# --- IPDK 연동 지점 ---' 부분을 채우십시오.

버전: 1.0.0   /   측정 방법: 세로 라인 프로파일 + 50% 크로싱(선형 보간)
"""

import numpy as np

VERSION = "1.0.0"

# --- 설정(파라미터): 코드와 분리 ---
DEFAULT_PARAMS = {
    "n_positions": 20,     # 측정할 스캔 위치 개수(균등 분포)
    "smooth": 7,           # 프로파일 평활 창(홀수). 0이면 미적용
    "min_edges": 2,        # 유효 측정으로 인정할 최소 엣지 개수
}


def _profile(image, x, smooth):
    col = image[:, x].astype(float)
    if smooth and smooth >= 3:
        pad = smooth // 2
        cp = np.pad(col, pad, mode="edge")          # 가장자리 값으로 패딩(0 패딩 금지)
        col = np.convolve(cp, np.ones(smooth) / smooth, mode="valid")
    return col


def _edges_50pct(prof):
    """50% 레벨을 지나는 위치들을 선형 보간으로 반환(서브픽셀)."""
    level = (np.percentile(prof, 5) + np.percentile(prof, 95)) / 2.0   # robust 50% 레벨
    xs = []
    for i in range(len(prof) - 1):
        y0, y1 = prof[i], prof[i + 1]
        if (y0 - level) * (y1 - level) < 0:
            t = (level - y0) / (y1 - y0)
            xs.append(i + t)
    return xs


def measure(image, scale, params=None):
    """
    이미지에서 밝은 층의 두께를 여러 위치에서 재고 통계를 반환한다.

    입력
      image  : 2D uint8 grayscale ndarray
      scale  : nm/pixel (Phase 4 에서 파일로부터 읽은 값)
      params : 설정 dict (없으면 DEFAULT_PARAMS)

    반환 (dict)
      성공: {'mean_nm','std_nm','n','unit','version','status':'ok'}
      실패: {'status':'fail','error': <사유>, 'version':...}
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    try:
        if image is None or image.ndim != 2:
            return {"status": "fail", "error": "2D grayscale 이미지가 아님", "version": VERSION}

        H, W = image.shape
        xs = np.linspace(int(W * 0.1), int(W * 0.9), p["n_positions"]).astype(int)

        results = []
        for x in xs:
            prof = _profile(image, x, p["smooth"])
            edges = _edges_50pct(prof)
            if len(edges) < p["min_edges"]:
                continue                      # 측정 실패는 건너뜀
            results.append((edges[-1] - edges[0]) * scale)

        if not results:
            return {"status": "fail", "error": "유효 측정 없음", "version": VERSION}

        results = np.array(results)
        return {
            "mean_nm": float(results.mean()),
            "std_nm": float(results.std()),
            "n": int(results.size),
            "unit": "nm",
            "version": VERSION,
            "status": "ok",
        }
    except Exception as e:                    # 어떤 오류에도 멈추지 않음
        return {"status": "fail", "error": str(e), "version": VERSION}


# --- 로컬 테스트: 배포 전 재현성 확인 ---
def _self_test():
    rng = np.random.default_rng(3)
    img = np.full((220, 360), 90, float)
    img[95:130, :] = 185                      # 참 두께 35 px
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

    r = measure(img, scale=0.25)
    assert r["status"] == "ok", r
    assert abs(r["mean_nm"] - 8.75) < 0.3, r          # 알려진 정답과 대조
    assert measure(img, 0.25) == r                    # 같은 입력 -> 같은 출력
    assert measure(None, 0.25)["status"] == "fail"    # 실패 입력도 안전
    print("self-test 통과:", r)


if __name__ == "__main__":
    _self_test()


# --- IPDK 연동 지점 (사내 문서에 맞춰 채우기) ---
# IPDK 가 이미지·scale·파라미터를 넘겨주는 방식에 맞춰 아래를 구현하십시오.
#
# def ipdk_entry(context):
#     image  = context.get_image()        # IPDK 입력 후크
#     scale  = context.get_scale()        # 메타데이터에서 scale
#     params = context.get_params()       # 레시피 파라미터
#     result = measure(image, scale, params)
#     context.report(result)              # IPDK 출력 후크
