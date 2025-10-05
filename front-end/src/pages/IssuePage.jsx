import emailjs from "@emailjs/browser";
import { useState } from "react";

function IssuePage() {
  const [feedback, setFeedback] = useState("");
  const [userEmail, setUserEmail] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);

  function sendMail() {
    if (!feedback || !userEmail) {
      alert("Vui lòng nhập đầy đủ nội dung phản hồi và email của bạn.");
      return;
    }

    let templateParams = {
      from_name: userEmail, // Email của người gửi
      message: feedback, // Nội dung phản hồi
      reply_to: userEmail, // Email để phản hồi lại
    };

    emailjs
      .send(
        "service_3pur135", // Thay bằng Service ID của bạn
        "template_pxhe9ab", // Thay bằng Template ID của bạn
        templateParams,
        "2wVBISFW8_j42sOUf" // Thay bằng Public Key của bạn
      )
      .then(
        function (response) {
          console.log("SUCCESS!", response.status, response.text);
          setIsModalOpen(true);
          setFeedback("");
          setUserEmail("");
        },
        function (error) {
          console.log("FAILED...", error);
          alert("Đã xảy ra lỗi. Vui lòng thử lại sau.");
          setFeedback("");
          setUserEmail("");
        }
      );
  }

  return (
    <div className="flex justify-center h-[85vh] bg-gradient-to-br from-blue-100 to-purple-100">
      {/* Modal */}
      {isModalOpen && (
        <div className="modal modal-open">
          <div className="modal-box">
            <h3 className="font-bold text-lg">Gửi thành công 🥳</h3>
            <p className="py-4">
              Cảm ơn bạn đã gửi góp ý / báo lỗi 🤗. Chúng tôi sẽ xem xét những ý
              kiến của người dùng để ngày càng hoàn thiện sản phẩm hơn nhé!
            </p>
            <div className="modal-action">
              <button
                onClick={() => setIsModalOpen(false)}
                className="btn btn-success"
              >
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="md:w-[50%]">
        <h1 className="text-3xl text-center font-bold p-5 bg-[linear-gradient(90deg,hsl(var(--s))_0%,hsl(var(--sf))_9%,hsl(var(--pf))_42%,hsl(var(--p))_47%,hsl(var(--a))_100%)] bg-clip-text will-change-auto [-webkit-text-fill-color:transparent] [transform:translate3d(0,0,0)] motion-reduce:!tracking-normal max-[1280px]:!tracking-normal [@supports(color:oklch(0_0_0))]:bg-[linear-gradient(90deg,hsl(var(--s))_4%,color-mix(in_oklch,hsl(var(--sf)),hsl(var(--pf)))_22%,hsl(var(--p))_45%,color-mix(in_oklch,hsl(var(--p)),hsl(var(--a)))_67%,hsl(var(--a))_100.2%)]">
          Báo lỗi hoặc góp ý
        </h1>
        <p className="text-justify font-semibold text-sm pr-2 pl-2">
          Sự đóng góp ý kiến từ các bạn sẽ là sự hỗ trợ đắc lực giúp chúng tôi
          ngày càng hoàn thiện sản phẩm hơn.
        </p>

        <textarea
          placeholder="Nhập phản hồi của bạn tại đây!"
          className="mt-5 mb-3 h-[30%] textarea textarea-bordered textarea-md w-full"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
        ></textarea>

        <input
          type="email"
          placeholder="Email của bạn"
          className="input w-full max-w-xs"
          value={userEmail}
          onChange={(e) => setUserEmail(e.target.value)}
        />

        <button
          onClick={sendMail}
          className="mt-5 w-full btn btn-primary btn-md bg-gradient-to-tl from-transparent via-blue-600 to-indigo-500"
        >
          Gửi ý kiến
        </button>
      </div>
    </div>
  );
}

export default IssuePage;
