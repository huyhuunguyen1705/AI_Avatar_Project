# Dự án AI Avatar Cá Nhân Hóa (Personalized AI Avatar Project)

Chào mừng bạn đến với Dự án AI Avatar Cá Nhân Hóa! Dự án này khám phá và triển khai các kỹ thuật tiên tiến trong lĩnh vực tạo ảnh AI để tạo ra các avatar độc đáo và cá nhân hóa từ ảnh của người dùng. Chúng tôi tập trung vào việc sử dụng các phương pháp như Textual Inversion, Dreambooth, LoRA, và phương pháp one-shot InstantID.


## Giới thiệu

Trong thời đại kỹ thuật số, avatar đóng vai trò quan trọng trong việc thể hiện bản sắc cá nhân trực tuyến. Dự án này nhằm mục đích cung cấp một bộ công cụ và quy trình để người dùng có thể tạo ra các avatar AI chất lượng cao, mang đậm dấu ấn cá nhân của riêng họ hoặc của một chủ thể cụ thể. Chúng tôi khám phá sự cân bằng giữa chất lượng, tính linh hoạt, và yêu cầu về tài nguyên tính toán của các phương pháp khác nhau.

## Tính năng nổi bật

*   **Cá nhân hóa cao**: Tạo avatar dựa trên đặc điểm khuôn mặt từ ảnh đầu vào.
*   **Đa dạng phong cách**: Áp dụng nhiều phong cách nghệ thuật khác nhau cho avatar.
*   **Linh hoạt trong huấn luyện**: Hỗ trợ nhiều kỹ thuật fine-tuning và one-shot.
*   **Tối ưu hóa tài nguyên**: Sử dụng LoRA để giảm thiểu chi phí huấn luyện và lưu trữ.
*   **Tạo ảnh nhanh chóng**: InstantID cho phép tạo avatar từ một ảnh duy nhất.

## Các công nghệ sử dụng

Dự án này tích hợp và so sánh hiệu quả của các phương pháp sau:

### Textual Inversion

*   **Mô tả**: Textual Inversion là một kỹ thuật cho phép chúng ta "dạy" cho một mô hình khuếch tán (diffusion model) một khái niệm mới (ví dụ: một người cụ thể, một đối tượng, một phong cách) từ một vài hình ảnh ví dụ. Nó thực hiện điều này bằng cách tối ưu hóa một từ nhúng (word embedding) mới trong không gian nhúng văn bản của mô hình, đại diện cho khái niệm đó.
*   **Ưu điểm**: Tương đối nhẹ, file embedding nhỏ, có thể kết hợp nhiều embedding.
*   **Nhược điểm**: Có thể không nắm bắt được các chi tiết phức tạp bằng Dreambooth.
*   **Ứng dụng trong dự án**: Tạo một "từ khóa" đại diện cho người dùng để tạo avatar.

### Dreambooth

*   **Mô tả**: Dreambooth là một phương pháp fine-tuning toàn bộ hoặc một phần của mô hình khuếch tán (thường là UNet) để "ghi nhớ" một chủ thể cụ thể từ một tập hợp nhỏ các hình ảnh. Nó sử dụng một định danh lớp (class identifier) duy nhất để phân biệt chủ thể.
*   **Ưu điểm**: Cho phép tái tạo chủ thể với độ trung thực cao và linh hoạt trong nhiều bối cảnh, phong cách.
*   **Nhược điểm**: Yêu cầu tài nguyên tính toán lớn hơn và thời gian huấn luyện lâu hơn so với Textual Inversion. File model lớn.
*   **Ứng dụng trong dự án**: Huấn luyện mô hình chuyên biệt cho một người dùng để tạo avatar chất lượng cao.

### LoRA (Low-Rank Adaptation)

*   **Mô tả**: LoRA là một kỹ thuật hiệu quả để fine-tuning các mô hình ngôn ngữ lớn (và các mô hình transformer nói chung, bao gồm cả các thành phần trong diffusion models). Thay vì fine-tuning toàn bộ trọng số của mô hình, LoRA chỉ huấn luyện một số lượng nhỏ các ma trận có hạng thấp (low-rank matrices) được thêm vào các lớp hiện có.
*   **Ưu điểm**: Giảm đáng kể số lượng tham số cần huấn luyện, file model nhỏ hơn nhiều so với Dreambooth đầy đủ, huấn luyện nhanh hơn, dễ dàng kết hợp nhiều LoRA.
*   **Nhược điểm**: Có thể không đạt được độ trung thực tuyệt đối như Dreambooth full fine-tuning trong một số trường hợp phức tạp.
*   **Ứng dụng trong dự án**: Fine-tuning hiệu quả cho các chủ thể hoặc phong cách cụ thể, dễ dàng chia sẻ và sử dụng. Có thể áp dụng LoRA *cho* Dreambooth để giảm tài nguyên.

### InstantID (One-shot)

*   **Mô tả**: InstantID là một phương pháp "one-shot" tiên tiến, cho phép tạo ra hình ảnh cá nhân hóa chất lượng cao từ một hình ảnh tham chiếu duy nhất. Nó tập trung vào việc duy trì tính nhất quán của danh tính (ID consistency) trong khi vẫn cho phép điều khiển phong cách và tư thế thông qua prompt văn bản.
*   **Ưu điểm**: Chỉ cần một ảnh đầu vào, nhanh chóng, duy trì danh tính tốt.
*   **Nhược điểm**: Chất lượng và sự linh hoạt có thể phụ thuộc nhiều vào chất lượng ảnh đầu vào và khả năng của mô hình nền.
*   **Ứng dụng trong dự án**: Tạo avatar nhanh chóng khi người dùng chỉ có một ảnh chân dung.

## Workflow đề xuất

1.  **Thử nghiệm nhanh (One-shot):**
    *   Sử dụng **InstantID** với một ảnh duy nhất để có cái nhìn tổng quan nhanh về khả năng tạo avatar.
    *   Đánh giá độ tương đồng và chất lượng.

2.  **Huấn luyện nhẹ (Textual Inversion):**
    *   Nếu cần kiểm soát tốt hơn và có một vài ảnh (5-10), huấn luyện một embedding **Textual Inversion**.
    *   Thử nghiệm với các prompt khác nhau.

3.  **Huấn luyện chất lượng cao (Dreambooth/LoRA):**
    *   Nếu yêu cầu độ trung thực và linh hoạt cao nhất, và có đủ dữ liệu (10-20+ ảnh):
        *   Huấn luyện một mô hình **LoRA**: Cân bằng tốt giữa chất lượng, tốc độ huấn luyện và kích thước file. Đây thường là lựa chọn tốt cho hầu hết các trường hợp.
        *   Huấn luyện một mô hình **Dreambooth** (Dreambooth - LoRA): Khi cần độ chính xác tối đa và có tài nguyên tính toán.
    *   So sánh kết quả giữa LoRA và Dreambooth.

4.  **Kết hợp:**
    *   Sử dụng một model Dreambooth/LoRA làm nền cho chủ thể và kết hợp với các LoRA phong cách khác để tăng tính đa dạng.
    *   Sử dụng embedding Textual Inversion cùng với các model Dreambooth/LoRA (nếu tương thích).

## Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng xem qua `CONTRIBUTING.md` (nếu có) để biết cách tham gia.
Các cách bạn có thể đóng góp:
*   Báo cáo lỗi (bug report)
*   Yêu cầu tính năng mới (feature request)
*   Gửi Pull Request với các cải tiến hoặc sửa lỗi
*   Chia sẻ các kết quả huấn luyện và prompt thú vị

## Hướng phát triển tương lai

*   [ ] Xây dựng giao diện người dùng (Web UI) thân thiện.
*   [ ] Tích hợp thêm các phương pháp tạo avatar mới.
*   [ ] Tối ưu hóa quy trình huấn luyện và tạo ảnh.
*   [ ] Nghiên cứu về việc kiểm soát biểu cảm, tư thế tốt hơn.
*   [ ] Hỗ trợ tạo video avatar ngắn.