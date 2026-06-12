import { useCallback, useEffect, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Minus, Square, X } from "lucide-react";
import chinaMobileLogo from "./assets/china-mobile-logo.svg";

type PageItem = { id: string; label: string; icon: React.ReactNode };

function safeCurrentWindow() {
  try {
    return getCurrentWindow();
  } catch {
    return null;
  }
}

export function TitleBar({
  pages,
  currentPage,
  onPageChange,
}: {
  pages: PageItem[];
  currentPage: string;
  onPageChange: (id: string) => void;
}) {
  const [maximized, setMaximized] = useState(false);
  const draggingRef = useRef(false);

  useEffect(() => {
    const win = safeCurrentWindow();
    if (!win) return;
    win.isMaximized().then(setMaximized).catch(() => {});

    let unlisten: (() => void) | null = null;
    win
      .onResized(() => {
        win.isMaximized().then(setMaximized).catch(() => {});
      })
      .then((fn) => {
        unlisten = fn;
      })
      .catch(() => {});

    return () => {
      unlisten?.();
    };
  }, []);

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest("button")) return;
    draggingRef.current = true;
    safeCurrentWindow()?.startDragging().catch(() => {}).finally(() => {
      draggingRef.current = false;
    });
  }, []);

  const handleMinimize = useCallback(() => {
    safeCurrentWindow()?.minimize().catch(() => {});
  }, []);

  const handleToggleMaximize = useCallback(() => {
    safeCurrentWindow()?.toggleMaximize().catch(() => {});
  }, []);

  const handleClose = useCallback(() => {
    safeCurrentWindow()?.close().catch(() => {});
  }, []);

  return (
    <div className="titlebar" onPointerDown={handlePointerDown}>
      <div className="titlebar-left">
        <img className="titlebar-logo" src={chinaMobileLogo} alt="中国移动" />
        <span className="titlebar-appname">手眼标定</span>
        <nav className="titlebar-tabs" aria-label="主导航" role="tablist">
          {pages.map((item) => (
            <button
              key={item.id}
              className={`titlebar-tab${currentPage === item.id ? " active" : ""}`}
              onClick={() => onPageChange(item.id)}
              role="tab"
              aria-selected={currentPage === item.id}
              type="button"
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
      </div>
      <div className="titlebar-right">
        <button
          className="titlebar-btn titlebar-minimize"
          onClick={handleMinimize}
          aria-label="最小化"
          type="button"
        >
          <Minus size={14} className="titlebar-btn-icon" />
        </button>
        <button
          className="titlebar-btn titlebar-maximize"
          onClick={handleToggleMaximize}
          aria-label={maximized ? "还原" : "最大化"}
          type="button"
        >
          <Square size={12} className="titlebar-btn-icon" />
        </button>
        <button
          className="titlebar-btn titlebar-close"
          onClick={handleClose}
          aria-label="关闭"
          type="button"
        >
          <X size={14} className="titlebar-btn-icon" />
        </button>
      </div>
    </div>
  );
}
