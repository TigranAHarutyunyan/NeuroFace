# ✅ Այս կոդը YOLO մոդելի միջոցով վիդեոից դեմքերի ավտոմատ batch-ով հայտնաբերման և պահպանման pipeline է:

# Այն աշխատում է այս flow-ով:
# VideoProcessor.process() => _split_video_ranges() => process_range() => process_batch()
# 
# ✅ Մուտքային ինֆո
#   - video_path (տեսանյութի ուղի)
#   - model_path (YOLO մոդելի ուղի)
#   - frame_skip (քանի ֆրեյմը մեկ վերցնենք)
#   - output_dir (որտեղ պահպանել դեմքերը)

# ✅ Օգտագործման օրինակ:
# from your_module import VideoProcessor
# 
# processor = VideoProcessor(
#     video_path="path/to/video.mp4",
#     model_path="models/yolov8n-face.pt",
#     frame_skip=5,
#     output_dir="./output/faces",
#     max_gpu_workers=1  # կամ ավել, եթե VRAM թույլ է տալիս
# )
# 
# processor.process()

# ✅ Ինչ է իրականում կատարվում
# - Վիդեոն բաժանվում է մասերի (chunks), ըստ չափի ու աշխատող պրոցեսորների
# - Յուրաքանչյուր մասից ֆրեյմեր կարդացվում են FrameDataset class-ի միջոցով
# - DataLoader-ն փոխանցում է ֆրեյմերը YOLO մոդելին batch-երով
# - Յուրաքանչյուր ֆրեյմի վրա YOLO-ն հայտնաբերում է դեմքեր (bbox)
# - Դեմքերը կտրում ենք, պահպանում JPEG ֆորմատով
# - Ամեն բանի usage-ն (RAM/GPU) լոգվում է
# - ProgressBar-ն live ցուցադրում է ընթացքում առաջընթացը

# ✅ Եթե վիդեոն մեծ է (մինչև կամ ավելի քան 100MB)
# - Ավտոմատ աշխատում է sequential եղանակով (1 պրոցես)
# - Batch size-ը նվազում է VRAM-ի հիման վրա (օր. 1GB RAM => batch_size=1)

# ✅ Եթե վիդեոն փոքր է
# - Օգտագործվում է մի քանի պրոցեսով parallel inference
# - GPU և RAM բեռը ավելի լավ բաշխվում են

# ✅ Բոլոր պրոցեսները գրանցում են
# - Քանի ֆրեյմ մշակվեց
# - Քանի դեմք հայտնաբերվեց
# - Որքան RAM և GPU օգտագործվեց

# Այս կոդը արդեն լրիվ օգտագործելի է արտադրական պայմաններում՝ YOLO հիմքով video-face detection լուծման համար
# Դու կարող ես այն ինտեգրել նաև clustering-ի, enhancement-ի կամ web dashboard-ի հետ

