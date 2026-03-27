const toBoolean = (value, defaultValue = false) => {
  if (value === undefined) {
    return defaultValue;
  }

  return String(value).toLowerCase() === "true";
};

export const appConfig = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
  requestTimeoutSeconds: Number(import.meta.env.VITE_REQUEST_TIMEOUT_SECONDS || 60),
  ngrokBypassHeader: toBoolean(import.meta.env.VITE_ENABLE_NGROK_BYPASS_HEADER, true),
  emailServiceId: import.meta.env.VITE_EMAILJS_SERVICE_ID || "",
  emailTemplateId: import.meta.env.VITE_EMAILJS_TEMPLATE_ID || "",
  emailPublicKey: import.meta.env.VITE_EMAILJS_PUBLIC_KEY || "",
};
