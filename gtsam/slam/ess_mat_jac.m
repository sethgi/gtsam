clear; clc;

%% Symbolic definitions
syms tx ty tz real
syms r11 r12 r13 r21 r22 r23 r31 r32 r33 real
assumeAlso([r11 r12 r13 r21 r22 r23 r31 r32 r33],'real');

% Rotation matrix (aRb)
aRb = [r11 r12 r13;
       r21 r22 r23;
       r31 r32 r33];

% Translation vector (aTb)
aTb = [tx; ty; tz];

%% Unit3 direction and its Jacobian wrt translation
norm_t = sqrt(tx^2 + ty^2 + tz^2);
direction = aTb / norm_t;

% 3x3 derivative of normalized vector wrt translation
D_unit_3x3 = simplify(jacobian(direction, [tx ty tz]));

%% Construct a symbolic, fixed tangent basis orthogonal to direction
% We'll choose an arbitrary reference axis and project it to the tangent plane
ref = [1; 0; 0];
proj = ref - (direction.' * ref) * direction;
b1 = simplify(proj / norm(proj));
b2 = simplify(cross(direction, b1));  % guaranteed orthogonal

B = simplify([b1 b2]);  % 3x2 tangent basis

% 2x3 Jacobian of tangent-space coordinates wrt translation
D_dir_2x3 = simplify(B.' * D_unit_3x3);

%% Assemble final 5x6 Jacobian as in GTSAM::FromPose3
I3 = eye(3);
Z3 = zeros(3);
Z23 = zeros(2,3);

H = simplify([ I3 Z3;
               Z23 D_dir_2x3 * aRb ]);

disp('H (5x6) = d(E parameters) / d(Pose3 parameters):');
pretty(H)
