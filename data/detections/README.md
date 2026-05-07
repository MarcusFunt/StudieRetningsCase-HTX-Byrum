# Detection CSV Files

Save USB-serial bbox CSV files here, for example `session.csv`.

Expected columns:

```text
timestamp_ms,frame_id,detection_id,bbox_x,bbox_y,bbox_w,bbox_h,confidence,target
```

These files contain anonymous detection geometry only. They must not contain image data, video frames, or camera stream URLs.
