/** 用系统 AppKit 显示一张登录二维码，窗口生命周期归启动器所有。 */
ObjC.import('AppKit')

/** 图片由上游登录进程生成，窗口只接收本机临时文件路径。 */
function run([imagePath]) {
  const image = $.NSImage.alloc.initWithContentsOfFile(imagePath)
  if (image.isNil()) throw new Error('无法读取微信登录二维码。')

  const app = $.NSApplication.sharedApplication
  app.setActivationPolicy($.NSApplicationActivationPolicyAccessory)

  const window = $.NSWindow.alloc.initWithContentRectStyleMaskBackingDefer(
    $.NSMakeRect(0, 0, 360, 382),
    $.NSWindowStyleMaskTitled | $.NSWindowStyleMaskClosable,
    $.NSBackingStoreBuffered,
    false,
  )
  window.title = '微信扫码登录'
  window.releasedWhenClosed = false

  const instructions = $.NSTextField.labelWithString('请用手机微信扫码并确认登录')
  instructions.frame = $.NSMakeRect(24, 340, 312, 24)
  instructions.alignment = $.NSTextAlignmentCenter
  window.contentView.addSubview(instructions)

  const qr = $.NSImageView.alloc.initWithFrame($.NSMakeRect(24, 16, 312, 312))
  qr.image = image
  qr.imageScaling = $.NSImageScaleProportionallyUpOrDown
  window.contentView.addSubview(qr)

  window.center
  window.makeKeyAndOrderFront(null)
  app.activateIgnoringOtherApps(true)
  app.run
}
