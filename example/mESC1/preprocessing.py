
import scanpy as sc
import pandas as pd
from statsmodels.stats.multitest import multipletests

# 데이터 불러오기 (GSE98664 파일을 읽어들임)
file_path = '/home/younggun0816/GRN/GRACE/example/mESC1/GSE98664_tpm_sailfish_mergedGTF_RamDA_mESC_differentiation_time_course.txt'
data = pd.read_csv(file_path, sep='\t', index_col=0)  # 파일 형식에 따라 구분자 지정.

# 숫자형 데이터와 문자열 데이터 분리
numeric_data = data.drop(columns=['transcript_id']).values.astype(np.float64)  # 숫자 데이터만 추출
transcript_ids = data['transcript_id'].values  # 문자열 데이터 추출
numeric_data = numeric_data.T


adata = sc.AnnData(X=numeric_data)
# transcript_id를 adata.obs에 추가
adata.obs['transcript_id'] = transcript_ids

# 1. 품질 제어(Quality Control)
# 90% 이상의 세포에서 발현되지 않은 유전자 필터링
sc.pp.filter_genes(adata, min_cells=int(0.1 * adata.shape[0]))
sc.pp.filter_cells(adata, min_genes=int(0.1 * adata.shape[1]))  


# 2. 정규화 및 변환
# CPM 정규화 (Scanpy는 기본적으로 counts per million을 적용함)
adata_CPM = sc.pp.normalize_total(adata, target_sum=1e6, copy=True)
# 로그 변환
sc.pp.log1p(adata)

## 이미 TPM이 되어있는 파일이기때문에, TPM은 따로 log1p만 거친다.
adata_TPM = sc.pp.log1p(adata, copy=True)

# 3. 변동성 높은 유전자 선택
# 변동성이 큰 상위 유전자 선택, Bonferroni 보정을 적용하여 p-값 기준 선택
sc.pp.highly_variable_genes(adata_CPM, n_top_genes=1000, flavor='seurat_v3', batch_key=None)
sc.pp.highly_variable_genes(adata_TPM, n_top_genes=1000, flavor='seurat_v3', batch_key=None)


# 선택된 변동성 높은 유전자와 그들의 p-value 추출
hv_genes = adata.var[adata.var['highly_variable']]


# 필터링된 데이터(유전자와 세포 모두 선택된) DataFrame으로 변환 후 CSV로 저장
filtered_data = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
filtered_data.to_csv('/home/younggun0816/GRN/GRACE/example/mESC1/filtered_data.csv')
