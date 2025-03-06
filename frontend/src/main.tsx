import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./index.css";
import App from "./App";
import GitHubRagExample from "./pages/GitHubRagExample";

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  {
    path: "/github-rag",
    element: <GitHubRagExample />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
