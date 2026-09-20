// ============================================================
//  ZhiGuo E-Nose Chamber Box  (v7)
//  Sealed gas measurement box. For photos, lift the lid and
//  use your own phone stand at a fixed height/position.
//  White tray + color-card recess keeps photo color stable.
//
//  Parts (export STL one by one):
//    sealed_box()    - big open-top box (device + sample)
//    sliding_lid()   - lid with sliding acrylic window
//    slide_stop()    - end stop glued on the lid (acrylic slide-in)
//    usb_plug()      - plug for the USB hole
//    sample_tray()   - white tray + color card recess
//
//  Acrylic sheet to buy: 3mm, cut to panel_w x panel_d
//  Free software: https://openscad.org
// ============================================================

$fn = 60;

// ---------- BOX ----------
box_w = 365;   // outer width X (device is 350mm long)
box_d = 170;   // outer depth Y (device is 150mm wide)
box_h = 230;   // outer height Z (device ~200mm + lid lip room)

wall   = 2.4;
floor  = 2.4;

// ---------- SLIDING WINDOW LID ----------
lid_t = 4;            // lid plate thickness
slide_t = 3;          // acrylic sheet thickness (buy 3mm)
slide_tol = 0.6;      // acrylic clearance inside the pocket
win_w = 200;          // window width X
win_d = 90;           // window depth Y
slide_travel = 80;    // how far the acrylic slides to open
overlap = 5;          // acrylic overlap over window when closed
rail_m = 6;           // margin from window edge to pocket wall
wall_h = slide_t + 3; // pocket wall height above the plate
tol = 0.4;            // clearance between lid lip and box
lip_d = 8;            // lid lip depth into the box

// ---------- USB / CABLE ----------
usb_w = 22; usb_h = 12; usb_center_z = 30;   // big enough for the USB cable
plug_w = 24; plug_h = 14; plug_l = 14;

// ---------- MISC ----------
post_d = 6; post_h = 6;

// ---------- COMPUTED ----------
inner_w = box_w - 2*wall;
inner_d = box_d - 2*wall;
inner_h = box_h - floor;

// sliding lid geometry
Px0 = (box_w - (win_w + slide_travel + 2*rail_m)) / 2;
Px1 = Px0 + win_w + slide_travel + 2*rail_m;
Py0 = (box_d - (win_d + 2*rail_m)) / 2;
Py1 = Py0 + win_d + 2*rail_m;
Wx0 = Px0 + rail_m;   // window left edge
Wy0 = Py0 + rail_m;   // window front edge
panel_w = win_w + 2*overlap;               // acrylic length to cut
panel_d = win_d + 2*rail_m - 2*slide_tol;  // acrylic width to cut

module sealed_box() {
  difference() {
    cube([box_w, box_d, box_h]);
    // cavity (open top)
    translate([wall, wall, floor])
      cube([inner_w, inner_d, inner_h - floor]);
    // rear USB hole
    translate([box_w/2, box_d - wall - 1, usb_center_z])
      cube([usb_w, wall + 2, usb_h], center=true);
    // breadboard posts
    for (i=[0:1]) for (j=[0:1]) {
      translate([12 + i*(inner_w - 24), 12 + j*(inner_d - 24), floor])
        cylinder(d = post_d, h = post_h);
    }
  }
}

module sliding_lid() {
  difference() {
    union() {
      // base plate
      cube([box_w, box_d, lid_t]);
      // alignment lip (hollow rim, saves plastic)
      translate([wall + tol, wall + tol, -lip_d])
        difference() {
          cube([inner_w - 2*tol, inner_d - 2*tol, lip_d]);
          translate([4, 4, -1])
            cube([inner_w - 2*tol - 8, inner_d - 2*tol - 8, lip_d + 2]);
        }
      // pocket walls: front, back, left, right
      translate([Px0, 0, lid_t]) cube([Px1 - Px0, Py0, wall_h]);
      translate([Px0, Py1, lid_t]) cube([Px1 - Px0, box_d - Py1, wall_h]);
      translate([Px1, Py0, lid_t]) cube([box_w - Px1, Py1 - Py0, wall_h]);
    }
    // window cut through the plate
    translate([Wx0, Wy0, -1]) cube([win_w, win_d, lid_t + 2]);
    // finger notch in the front wall (push the acrylic)
    translate([Wx0 - overlap - 5, -1, lid_t])
      cube([30, Py0 + 1, wall_h + 1]);
    // seal cord groove around the window (top of plate)
    translate([Wx0 - 3, Wy0 - 3, lid_t - 1.5])
      cube([win_w + 6, win_d + 6, 1.6]);
  }
}

module slide_stop() {
  // glue this at the left end of the lid AFTER sliding the acrylic in
  translate([0, Py0, lid_t])
    cube([Px0 + 2, Py1 - Py0, wall_h]);
}

module handle() {
  // glue this knob onto the acrylic sheet to slide it easily
  union() {
    cylinder(d = 14, h = 12);
    translate([0, 0, 12]) cylinder(d = 8, h = 8);
  }
}

module usb_plug() {
  cube([plug_w, plug_h, plug_l]);
}

module sample_tray() {
  tw = win_w - 40;
  td = win_d - 20;
  difference() {
    translate([0, 0, 3]) cube([tw, td, 5]);
    // shallow dish for the slices
    translate([6, 6, 5.5]) cube([tw - 12, td - 12, 3]);
    // color card recess (corner, glue a small color card in it)
    translate([8, 8, 5.5]) cube([24, 18, 3]);
  }
}

// ============================================================
//  PART SELECTOR (fool-proof)
//  Change `part` below to pick which part to export, then
//  press F6 (render) and F7 (export STL). Save each part
//  under its own file name.
//  Options: "sealed_box" "sliding_lid" "handle" "usb_plug"
//           "sample_tray" "slide_stop" "all"
// ============================================================
part = "sealed_box";

if (part == "sealed_box") {
  sealed_box();
} else if (part == "sliding_lid") {
  sliding_lid();
} else if (part == "handle") {
  handle();
} else if (part == "usb_plug") {
  usb_plug();
} else if (part == "sample_tray") {
  sample_tray();
} else {
  // all parts laid out side by side (one STL, print all at once)
  translate([box_w + 20, 0, 0]) sliding_lid();
  translate([0, box_d + 20, 0]) handle();
  translate([0, box_d + 60, 0]) usb_plug();
  translate([0, box_d + 100, 0]) sample_tray();
  translate([0, box_d + 140, 0]) slide_stop();
  sealed_box();
}