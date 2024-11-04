
import scanpy as sc
import pandas as pd


# 데이터 불러오기 (GSE98664 파일을 읽어들임)
file_path = '/home/younggun0816/GRN/GRACE/example/mESC1/GSE98664_tpm_sailfish_mergedGTF_RamDA_mESC_differentiation_time_course.txt'
data = pd.read_csv(file_path, sep='\t', index_col=0)  # 파일 형식에 따라 구분자 지정.

# Scanpy 객체로 변환
adata = sc.AnnData(data)


# 1. 품질 제어(Quality Control)
# 90% 이상의 세포에서 발현되지 않은 유전자 필터링
sc.pp.filter_genes(adata, min_cells=int(0.1 * adata.shape[0]))
sc.pp.filter_cells(adata, min_genes=10)  # 예시로 10개 이상의 유전자가 발현된 세포만 유지합니다.

# 2. 정규화 및 변환
# TPM 및 CPM 정규화 (Scanpy는 기본적으로 counts per million을 적용함)
sc.pp.normalize_total(adata, target_sum=1e6)
# 로그 변환
sc.pp.log1p(adata)

# 3. 변동성 높은 유전자 선택
# 변동성이 큰 상위 유전자 선택, Bonferroni 보정을 적용하여 p-값 기준 선택
sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat_v3', batch_key=None)

# 선택된 유전자만 남기기
adata = adata[:, adata.var.highly_variable]

# 결과 확인
print(adata)

# 필터링된 데이터(유전자와 세포 모두 선택된) DataFrame으로 변환 후 CSV로 저장
filtered_data = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
filtered_data.to_csv('/home/younggun0816/GRN/GRACE/example/mESC1/filtered_data.csv')
