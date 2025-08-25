G21         ; Set units to mm
G90         ; Absolute positioning

G1 X400 Y100 Z350 F500   ; Move to start (40cm, 10cm, 35cm)
G2 X450 Y150 I50 J0 Z350 ; CW arc: center at (450,100), end at (45,15)
G1 X500 Y150 Z350        ; Line to (50cm, 15cm, 35cm)
G1 X500 Y100 Z350        ; Close rectangle (optional)
G1 X400 Y100 Z350        ; Return to start
