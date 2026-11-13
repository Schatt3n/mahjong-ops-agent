import { useEffect, useRef, useState, type RefObject } from "react";
import type { Connection } from "./connection.ts";
import type { AttachmentDraft } from "./attachment-drafts.ts";
import { t } from "./i18n.ts";
import { Button, Icon } from "./ui.tsx";
import {
  bindComposerAttachments,
  accountAttachmentLimits,
  type ReviewedAttachment,
} from "./composer-attachments.ts";
import { ComposerHelp } from "./composer-help.tsx";
import { desktopCanCaptureRegion, desktopCaptureRegion } from "./desktop.ts";

export function AttachmentPicker({
  connection,
  account,
  threadId,
  enabled,
  mediaKinds,
  capabilities,
  composerRef,
  focusMessage,
}: {
  connection: Connection;
  account: string;
  threadId: string | null;
  enabled: boolean;
  mediaKinds: readonly string[];
  capabilities: Record<string, unknown>;
  composerRef: RefObject<HTMLDivElement | null>;
  focusMessage: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [error, setError] = useState(""),
    [dragging, setDragging] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const captureRequest = useRef<symbol | null>(null);
  function choose(files: File[]) {
    if (!enabled || !threadId) {
      setError(
        t("attachmentPicker.attachmentsArenTAvailableConversation"),
      );
      return;
    }
    try {
      connection.attachmentDrafts.add(
        account,
        threadId,
        files,
        mediaKinds,
        accountAttachmentLimits(capabilities),
      );
      setError("");
      focusMessage();
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : t("attachmentPicker.couldnTSelectTheseFiles"),
      );
    }
  }
  // 截图选择期间可能切换会话或退出账号。仅把结果放回发起截图时仍打开的草稿。
  async function capture() {
    if (!enabled || !threadId || captureRequest.current) return;
    const request = Symbol();
    captureRequest.current = request;
    setCapturing(true);
    setError("");
    const current = () => {
      const state = connection.session.snapshot();
      return captureRequest.current === request && state.account === account &&
        state.selected === threadId && !state.signInRequired;
    };
    try {
      const file = await desktopCaptureRegion();
      if (file && current()) choose([file]);
    } catch (error) {
      if (!current()) return;
      const code = error instanceof Error ? error.message : error;
      setError(
        code === "screen_capture_permission_denied"
          ? t("attachmentPicker.screenCapturePermission")
          : code === "screen_capture_too_large"
            ? t("attachmentPicker.screenCaptureTooLarge")
            : t("attachmentPicker.screenCaptureFailed"),
      );
    } finally {
      if (captureRequest.current === request) {
        captureRequest.current = null;
        setCapturing(false);
      }
    }
  }
  useEffect(() => () => { captureRequest.current = null; }, []);
  useEffect(() => {
    const root = composerRef.current;
    if (root)
      return bindComposerAttachments(root, choose, (value) =>
        setDragging(value && enabled && !!threadId),
      );
  }, [
    connection,
    account,
    threadId,
    enabled,
    mediaKinds,
    capabilities,
    composerRef,
  ]);
  useEffect(() => {
    if (!enabled) setDragging(false);
  }, [enabled]);
  return (
    <>
      <Button
        icon="attach"
        label={t("attachmentPicker.attachFile")}
        disabled={!enabled || !threadId}
        onClick={() => input.current?.click()}
      />
      {desktopCanCaptureRegion && mediaKinds.includes("image") && (
        <Button
          icon="capture-region"
          label={t("attachmentPicker.captureRegion")}
          disabled={!enabled || !threadId || capturing}
          onClick={() => { void capture(); }}
        />
      )}
      <input
        ref={input}
        className="attachment-input"
        type="file"
        multiple
        aria-label={t("attachmentPicker.chooseAttachment")}
        onChange={(event) => {
          const files = Array.from(event.target.files ?? []);
          event.target.value = "";
          if (files.length) choose(files);
        }}
      />
      {dragging && (
        <span className="composer-drop-hint" role="status">
          <Icon name="attach" /> {t("attachmentPicker.dropAttach")}
        </span>
      )}
      {error && (
        <span className="attachment-error" role="alert">
          {error}
        </span>
      )}
    </>
  );
}

export function AttachmentTray({
  draft,
  remove,
  quality,
}: {
  draft: AttachmentDraft;
  remove: (id: string) => void;
  quality: (original: boolean) => void;
}) {
  if (!draft.files.length) return null;
  return (
    <section
      className="composer-attachments"
      aria-label={t("attachmentPicker.stagedAttachments")}
    >
      <div className="composer-attachments-head">
        <span>
          {t("attachmentPicker.attachmentCount", { count: draft.files.length })}
        </span>
      </div>
      <ul className="composer-attachment-list">
        {draft.files.map((item) => (
          <li key={item.id} className="composer-attachment">
            <AttachmentPreview item={item} />
            <Button
              icon="close"
              label={t("attachmentPicker.remove", { name: item.file.name })}
              disabled={!!draft.submission}
              onClick={() => remove(item.id)}
            />
          </li>
        ))}
      </ul>
      {draft.files.some((item) => item.kind === "video") && (
        <div className="composer-original-video">
          <label>
            <input
              type="checkbox"
              checked={draft.videoQuality === "original"}
              disabled={!!draft.submission}
              onChange={(event) => quality(event.target.checked)}
            />
            {t("attachmentPicker.originalMediaQuality")}
          </label>
          <ComposerHelp
            label={t("attachmentPicker.aboutOriginalMediaQuality")}
          >
            {t("attachmentPicker.includeUnchangedVideoFileAlongside")}
          </ComposerHelp>
        </div>
      )}
    </section>
  );
}
function AttachmentPreview({
  item: { file, kind },
}: {
  item: ReviewedAttachment;
}) {
  const [url, setUrl] = useState(""),
    [previewFailed, setPreviewFailed] = useState(false);
  useEffect(() => {
    setPreviewFailed(false);
    if (kind === "file") return;
    const object = URL.createObjectURL(file);
    setUrl(object);
    return () => URL.revokeObjectURL(object);
  }, [file, kind]);
  const size =
    file.size < 1024 * 1024
      ? `${Math.ceil(file.size / 1024)} KB`
      : `${(file.size / 1024 / 1024).toFixed(1)} MB`;
  return (
    <>
      {url && kind === "image" && (
        <img
          className="attachment-review-preview"
          src={url}
          alt={t("attachmentPicker.stagedImage")}
        />
      )}
      {url && kind === "video" && !previewFailed && (
        <video
          className="attachment-review-preview"
          src={url}
          muted
          preload="metadata"
          aria-label={t("attachmentPicker.stagedVideo")}
          onError={() => setPreviewFailed(true)}
        />
      )}
      {kind === "video" && previewFailed && (
        <p className="field-help">
          {t("attachmentPicker.videoPreviewUnavailable")}
        </p>
      )}
      <div className="attachment-file">
        <Icon
          name={kind === "image" ? "image" : kind === "video" ? "play" : "file"}
        />
        <span>
          <strong>{file.name}</strong>
          <small>
            {kind === "file"
              ? t("attachmentPicker.file")
              : kind === "image"
                ? t("attachmentPicker.image")
                : t("attachmentPicker.video")}{" "}
            · {size}
          </small>
        </span>
      </div>
    </>
  );
}
