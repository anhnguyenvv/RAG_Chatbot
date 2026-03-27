import { appConfig } from "../config/env";

const buildHeaders = () => {
  if (!appConfig.ngrokBypassHeader) {
    return {};
  }

  return {
    "ngrok-skip-browser-warning": "69420",
  };
};

export async function askRag(source, question) {
  const endpoint = `${appConfig.apiBaseUrl}/rag/${source}?q=${encodeURIComponent(question)}`;
  const response = await fetch(endpoint, {
    method: "get",
    headers: buildHeaders(),
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}
