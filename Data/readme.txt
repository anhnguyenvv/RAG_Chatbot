Giới thiệu về tập dữ liệu
- Tập dữ liệu về Chương trình đào tạo Khoa Công nghệ thông tin, trường Đại học Khoa học Tự nhiên, ĐHQH - HCM (Chương trình Chuẩn).
- Mục đích thu thập dữ liệu: Xây dựng Chatbot RAG về Chương trình đào tạo FITHCMUS - Chương trình chuẩn
- Danh sách những người đóng góp đã tham gia vào dự án này:
	+ Đường Yến Ngọc
	+ Nguyễn Thị Lan Anh
	+ Nguyễn Thị Ngọc Châm 
	+ Huỳnh Phát Đạt
- Ngôn ngữ của tập dữ liệu: Tiếng Việt

Giới thiệu về Phương pháp thu thập dữ liệu
- Dữ liệu được thu thập và tổng hợp từ các nguồn:
	+ Đề cương môn học: Trang web khoa FITHCMUS > Đào tạo > Chương trình đào tạo > Chương trình Chuẩn (https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=36)
	+ Chương trình đào tạo: Trang web khoa FITHCMUS > Hệ thống sinh viên > CQuy > Chương trình đào tạo > CQuy Khóa tuyển 2023
	(https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=289)
	+ Câu hỏi và trả lời về Chương trình đào tạo: Trang Q&A FIT
        (https://courses.fit.hcmus.edu.vn/q2a/)
- Chuyển đổi dữ liệu thành dạng thuần văn bản:
	+ Định dạng dữ liệu theo cấu trúc chung (Đề mục, phân đoạn,...)
	+ Xử lý lỗi chính tả
	+ Chia thành các file .txt theo quy ước tên metadata-first:
	  <loai_tai_lieu>_nganh-<nganh>__khoa_<khoa_tuyen>__ban_hanh-<ngay>.txt
	  (bỏ qua các thành phần không xác định)
	Tên file	 Nội dung
	chuong_trinh_dao_tao_nganh-cong_nghe_thong_tin__khoa_2023__ban_hanh-2023-09-07.txt	- Chương trình đào tạo Ngành Công nghệ thông tin
	chuong_trinh_dao_tao_nganh-he_thong_thong_tin__khoa_2023__ban_hanh-2023-09-07.txt	- Chương trình đào tạo Ngành Hệ thống thông tin
	chuong_trinh_dao_tao_nganh-khoa_hoc_may_tinh__khoa_2023__ban_hanh-2023-09-07.txt	- Chương trình đào tạo Ngành Khoa học máy tính
	chuong_trinh_dao_tao_nganh-ky_thuat_phan_mem__khoa_2023__ban_hanh-2023-09-07.txt	- Chương trình đào tạo Ngành Kỹ thuật phần mềm
	chuong_trinh_dao_tao_nganh-tri_tue_nhan_tao__khoa_2023__ban_hanh-2023-09-07.txt	- Chương trình đào tạo Ngành Trí tuệ nhân tạo
	de_cuong_mon_hoc_nganh-da_nganh.txt	- Đề cương môn học các ngành/chuyên ngành
	dieu_kien_tot_nghiep_nganh-da_nganh.txt	- Điều kiện và quy trình thực hiện đề tài tốt nghiệp
	lien_thong_dai_hoc_thac_si_nganh-da_nganh.txt	- Danh sách môn học liên thông Đại học - Thạc sĩ
	quy_dinh_dao_tao_nganh-toan_truong__ban_hanh-2021-09-24.txt	- Quy định, quy chế của chương trình đào tạo
	quy_dinh_ngoai_ngu_nganh-toan_truong__khoa_2022__ban_hanh-2022-09-26.txt	- Quy định về Chuẩn đầu ra ngoại ngữ

Mô tả dữ liệu
Cách dữ liệu được tổ chức trong toàn bộ tập dữ liệu này:

Data/
	-Database/
		-chuong_trinh_dao_tao_nganh-cong_nghe_thong_tin__khoa_2023__ban_hanh-2023-09-07.txt
		-chuong_trinh_dao_tao_nganh-he_thong_thong_tin__khoa_2023__ban_hanh-2023-09-07.txt
		-chuong_trinh_dao_tao_nganh-khoa_hoc_may_tinh__khoa_2023__ban_hanh-2023-09-07.txt
		-chuong_trinh_dao_tao_nganh-ky_thuat_phan_mem__khoa_2023__ban_hanh-2023-09-07.txt
		-chuong_trinh_dao_tao_nganh-tri_tue_nhan_tao__khoa_2023__ban_hanh-2023-09-07.txt
		-de_cuong_mon_hoc_nganh-da_nganh.txt
		-dieu_kien_tot_nghiep_nganh-da_nganh.txt
		-lien_thong_dai_hoc_thac_si_nganh-da_nganh.txt
		-quy_dinh_dao_tao_nganh-toan_truong__ban_hanh-2021-09-24.txt
		-quy_dinh_ngoai_ngu_nganh-toan_truong__khoa_2022__ban_hanh-2022-09-26.txt
	-readme.txt

Liên kết Kho lưu trữ trực tuyến
https://www.kaggle.com/datasets/oiam2010/ctdt-fithcmus - Liên kết đến kho lưu trữ dữ liệu.

Giấy phép
Dự án này được cấp phép theo Giấy phép MIT - xem tệp LICENSE.md để biết chi tiết

