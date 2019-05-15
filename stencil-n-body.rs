// stencil-n-body.rs
// Copyright 2019 Timothée Jourde
//
// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You
// may obtain a copy of the License at
// 
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
// 
// DESCRIPTION
//   
//   This program performs a Barnes–Hut N-body simulation on a grid
//   together with a stencil code to produce an artistic video.
//   
// SETUP
//   
//   You need "rustc" (tested with version 1.34.0) to compile this
//   file, and "ffmpeg" (tested with version 4.1.1) must be in your
//   PATH.
//   
//   https://www.rust-lang.org/tools/install
//   https://ffmpeg.org/download.html
//   
// BUILD & RUN
//   
//   $ rustc -O stencil-n-body.rs
//   $ ./stencil-n-body 720 480 10 125 out.mp4
//                      ^   ^   ^  ^   ^
//                      1   2   3  4   5
//   
//   This will produce a raw RGB video and stream it to ffmpeg for
//   encoding; where (1) is the video width, (2) its height, (3) the
//   number of iterations to run before the video, (4) the number of
//   iterations/frames of the video and (5) the output file name
//   passed to ffmpeg. There are some other hard-coded parameters you
//   can tweak.

fn main() { app::app(); }

mod app {
  use std::io::Write;
  use std::process::{Command,Stdio,Child};

  use grid::Grid;

  struct Args<'a> {
    width:     usize,
    height:    usize,
    it_before: usize,
    it:        usize,
    out:       &'a String
  }

  pub fn app() {    
    let args: Vec<String> = std::env::args().collect();
    match parse_args(args.as_slice()) {
      None => eprintln!("usage: {} width height iterations-before \
                           iterations out.mp4", args[0]),
      
      Some(args) => {
        match Command::new("ffmpeg")
          .args(&["-video_size", &format!("{}x{}", args.width, args.height),
                  "-f", "rawvideo", "-pix_fmt", "rgb24", "-i", "-",
                  "-codec", "libx264", "-preset", "slow", "-crf", "18",
                  "-pix_fmt", "yuv420p", "-y", args.out])
          .stdin(Stdio::piped())
          .stdout(Stdio::null())
          .stderr(Stdio::null())
          .spawn() {
            Err(_)     => eprintln!("failed to start ffmpeg"),
            Ok(ffmpeg) => if run(args,ffmpeg).is_none() {
              eprintln!("\rffmpeg error                    ");
            }
          }
      }
    }
  }
  
  fn parse_args(args: &[String]) -> Option<Args> {
    match args {
      [_,a,b,c,d,e] => Some(Args {
        width:     a.parse().ok()?,
        height:    b.parse().ok()?,
        it_before: c.parse().ok()?,
        it:        d.parse().ok()?,
        out:       e }),
      _ => None
    }
  }
  
  fn run(args: Args, mut ffmpeg: Child) -> Option<()> {
    let print_it = |i| {
      print!("\r {: <5} / {: <5} ", i+1, args.it_before + args.it);
      std::io::stdout().flush().ok();
    };
    let ffmpeg = ffmpeg.stdin.as_mut()?;
    let mut grid = Grid::new(args.width, args.height);
    
    for i in 0 .. args.it_before {
      print_it(i);
      grid.iteration();
    }
    for i in 0 .. args.it {
      print_it(args.it_before+i);
      grid.iteration();
      ffmpeg.write(grid.as_rgb()).ok()?;
      ffmpeg.flush().ok()?;
    }
    println!();
    Some(())
  }
}

mod grid {
  use std::f32::consts::PI;

  use either::Either;
  use vec2::Vec2;
  use hash;
  use quadtree::Tree;

  struct Cell {
    charge:   f32,
    mass:     f32,
    velocity: Vec2<f32>
  }
  
  #[derive(Copy,Clone)]
  struct Dim { width: usize, height: usize }
  
  struct Node {
    charge: f32,
    mass:   f32,
    center: Vec2<f32>,
    leaf:   bool
  }
  
  pub struct Grid {
    matrix: Vec<Cell>,
    dim:    Dim,
    
    tree:       Tree<Node>,
    root_width: usize,
    
    time: usize,

    image: Vec<u8>,
    dirty: bool
  }
  
  impl Dim {
    fn size(self) -> usize {
      self.width * self.height
    }
    fn to(self, x: usize, y: usize) -> Option<usize> {
      if x >= self.width || y >= self.height { None }
      else { Some(y * self.width + x) }
    }
    fn ito(self, x: isize, y: isize) -> Option<usize> {
      if x < 0 || y < 0 { None }
      else { self.to(x as usize, y as usize) }
    }
    fn iter(self) -> impl Iterator<Item = (usize,usize)> {
      (0..self.size()).map(move |i| (i % self.width, i / self.width))
    }
  }

  impl Grid {
    const EPSILON: f32 = 0.0001;
    
    const THETA: f32 = 2.5;
    const G:     f32 = 3.5;
    
    const SPEED:      f32 = 0.4;
    const RESISTANCE: f32 = 8.0;
    
    const PERIOD: usize = 1000;
    
    const HUE_RANGE:  f32 = 315.0;
    const FROM_HUE:   f32 = 0.0;
    const SATURATION: f32 = 0.6;
    const BRIGHTNESS: f32 = 2.0;
    
    const DISCS:         usize = 222;
    const MIN_DISC_SIZE: f32   = 0.01;
    const MAX_DISC_SIZE: f32   = 0.1;
    const INIT_MASS:     f32   = 5.0;
    
    pub fn new(width: usize, height: usize) -> Grid {
      let dim = Dim {width: width, height: height};
      let root_width = width.max(height).next_power_of_two();

      let w  = width  as f32;
      let h  = height as f32;
      let lo = w.min(h) * Grid::MIN_DISC_SIZE / 2.0;
      let hi = w.min(h) * Grid::MAX_DISC_SIZE / 2.0;
      let discs: Vec<(Vec2<f32>,f32,f32)> = (0..Grid::DISCS).map(|i| {
        let r = lo + hash::noise(hash::pair(i,0)) * (hi-lo);
        let x = r + hash::noise(hash::pair(i,1)) * (w - 1.0 - 2.0*r);
        let y = r + hash::noise(hash::pair(i,2)) * (h - 1.0 - 2.0*r);
        (Vec2(x,y), r, hash::noise(hash::pair(i,3)))
      }).collect();
      
      let matrix: Vec<Cell> = dim.iter().map(|(x,y)| {
        let here = Vec2(x as f32, y as f32);
        let (cs,ms) = discs.iter().fold((0.0,0.0), |(cs,ms),(p,r,c)| {
          let m = 1.0 - ((here - *p).norm() / r).min(1.0);
          (cs + m*c, ms + m)
        });
        Cell {
          charge:   if ms > 0.0 {cs/ms} else {0.0},
          mass:     ms * Grid::INIT_MASS,
          velocity: Vec2::ZERO
        }
      }).collect();
      
      let tree = {
        let divide = |p| match p {
          (1,x,y) => {
            let cell = dim.to(x,y).and_then(|i| matrix.get(i));
            Either::Right( Node {
              charge: cell.map_or(0.0, |x| x.charge),
              mass:   cell.map_or(0.0, |x| x.mass),
              center: Vec2(x as f32, y as f32),
              leaf:   true
            })
          },
          (_,x,y) if x >= width || y >= height =>
            Either::Right( Node {
              charge: 0.0,
              mass:   0.0,
              center: Vec2(x as f32, y as f32),
              leaf:   true
            }),
          (w,x,y) => Either::Left((
            (w/2, x      , y      ),
            (w/2, x      , y + w/2),
            (w/2, x + w/2, y      ),
            (w/2, x + w/2, y + w/2)
          ))
        };
        Tree::unfold((root_width,0,0), &divide, &Grid::aggregate)
      };
      
      Grid {
        matrix: matrix,
        dim:    dim,
        
        tree:       tree,
        root_width: root_width,
        
        time: 0,

        image: vec![0; dim.size()*3],
        dirty: true
      }
    }

    pub fn as_rgb(&mut self) -> &[u8] {
      let image = self.image.as_mut_slice();
      if self.dirty {
        for (pixel,cell) in image.chunks_mut(3).zip(self.matrix.iter()) {
          let (r,g,b) = Grid::to_rgb(cell);
          pixel[0] = r;
          pixel[1] = g;
          pixel[2] = b;
        }
        self.dirty = false;
      }
      image
    }
    
    pub fn iteration(&mut self) {
      for (x,y) in self.dim.iter() {
        self.kernel(x,y);
      }
      self.update_tree();
      self.time  = (self.time + 1) % Grid::PERIOD;
      self.dirty = true;
    }
    
    fn kernel(&mut self, x: usize, y: usize) {
      let dim  = self.dim;
      dim.to(x,y).map(|i| {
        
        let (before,after) = self.matrix.as_mut_slice().split_at_mut(i);
        let (this,after) = after.split_at_mut(1);
        let this = &mut this[0];
        if this.mass > Grid::EPSILON {
          
          let polarity = self.time as f32 / Grid::PERIOD as f32;
          let polarity = - (polarity * 2.0 - 1.0).signum();
          let here = Vec2(x as f32, y as f32);
          
          let net_force = {
            let fold = |acc, node: &Node| {
              let charge_mass = polarity
                * Grid::charge_mass(node.charge, this.charge);
              let force = Grid::force( Grid::G, node.center, here,
                                       charge_mass.abs() * node.mass,
                                       charge_mass.abs() * this.mass );
              acc + force*charge_mass.signum()
            };
            let leaf = |width, node: &Node| node.leaf
              || width as f32 / (here - node.center).norm() < Grid::THETA;
            
            self.tree.fold(self.root_width, Vec2::ZERO, &fold, &leaf)
          };

          this.velocity = this.velocity + net_force;
          
          let mut drain = |dx: isize, dy: isize| {
            let dir     = Vec2(dx as f32, dy as f32).unit();
            let angle   = this.velocity.unit().dot(dir).acos();
            let quarter = PI / 4.0;
            if angle < quarter {
              dim.ito(x as isize + dx, y as isize + dy).map(|oi| {
                let mut other = if oi < i {
                  before.get_mut(oi)
                }
                else {
                  after.get_mut(oi-i-1)
                };
                other.as_mut().map(|other| {
                  let drained = (1.0 - angle/quarter) * this.velocity.norm();
                  let drained = drained.min(this.mass * Grid::SPEED);
                  let drained = drained.min(1.0 / (other.mass * Grid::RESISTANCE));

                  other.velocity = (other.velocity*other.mass + this.velocity*drained)
                    / (other.mass + drained);
                    
                  other.charge = (other.charge*other.mass + this.charge*drained)
                    / (other.mass + drained);
                    
                  this.mass  -= drained;
                  other.mass += drained;
                });
              });
            }
          };
          
          drain(-1, -1);
          drain(-1,  0);
          drain(-1,  1);
          drain( 0, -1);
          //     0,  0  
          drain( 0,  1);
          drain( 1, -1);
          drain( 1,  0);
          drain( 1,  1);
        }
      });
    }
    
    fn update_tree(&mut self) {
      let matrix = &self.matrix;
      let dim    = self.dim;
      let leaf = |x: &Node| {
        let cell = dim.to(x.center.0 as usize, x.center.1 as usize)
          .and_then(|i| matrix.get(i));
        Node {
          charge: cell.map_or(0.0, |c| c.charge),
          mass:   cell.map_or(0.0, |c| c.mass),
          .. *x
        }
      };
      self.tree.aggregate(&Grid::aggregate, &leaf);
    }
    
    fn aggregate(a: &Node, b: &Node, c: &Node, d: &Node) -> Node {
      let mass = a.mass + b.mass + c.mass + d.mass;
      let (charge,center) = if mass < Grid::EPSILON {
        (0.0, (a.center+b.center+c.center+d.center) / 4.0)
      }
      else {(
        ( a.charge * a.mass +
          b.charge * b.mass +
          c.charge * c.mass +
          d.charge * d.mass ) / mass,
        ( a.center * a.mass +
          b.center * b.mass +
          c.center * c.mass +
          d.center * d.mass ) / mass
      )};

      Node {
        charge: charge,
        mass:   mass,
        center: center,
        leaf:   mass < Grid::EPSILON
      }
    }
    
    // force applied on object 2 exerted by object 1
    fn force(g: f32, r1: Vec2<f32>, r2: Vec2<f32>, m1: f32, m2: f32) -> Vec2<f32> {
      let r = r2 - r1;
      let d = r.norm();
      if  d == 0.0 { Vec2::ZERO } else {
        r.unit() * (-g) * ((m2 * m1) / (d * d))
      }
    }
    
    // charge_mass > 0 means attraction, repulsion otherwise
    fn charge_mass(c1: f32, c2: f32) -> f32 {
      (c2 - c1).abs() * 2.0 - 1.0
    }

    fn to_rgb(cell: &Cell) -> (u8,u8,u8) {
      let hue   = (cell.charge * Grid::HUE_RANGE + Grid::FROM_HUE) % 360.0;
      let value = 1.0 - (1.0 / (cell.mass * Grid::BRIGHTNESS + 1.0));
      
      let f = |n| {
        let k: f32 = (n + hue/60.0) % 6.0;
        value - value * Grid::SATURATION
          * 0_f32.max( 1_f32.min( k.min( 4.0-k )))
      };
      let r = f(5.0) * 255.0;
      let g = f(3.0) * 255.0;
      let b = f(1.0) * 255.0;
      
      (r as u8, g as u8, b as u8)
    }
  }
}

mod quadtree {
  use either::Either;
  
  type Children<T> = (T,T,T,T);
  
  pub struct Tree<T> {
    value:    T,
    children: Option<Box<Children<Tree<T>>>>
  }
  
  impl<T> Tree<T> {
    pub fn unfold<P,D,C>(parent: P, divide: &D, aggregate: &C) -> Tree<T>
      where D: Fn(P) -> Either<Children<P>,T>,
            C: Fn(&T,&T,&T,&T) -> T {
      match divide(parent) {
        Either::Left((a,b,c,d)) => {
          let w = Tree::unfold(a,divide,aggregate);
          let x = Tree::unfold(b,divide,aggregate);
          let y = Tree::unfold(c,divide,aggregate);
          let z = Tree::unfold(d,divide,aggregate);
          let value = aggregate(&w.value, &x.value, &y.value, &z.value);
          Tree {
            value:    value,
            children: Some(Box::new((w,x,y,z)))
          }
        },
        Either::Right(v) => Tree {value: v, children: None}
      }
    }

    pub fn fold<A,F,L>(&self, width: usize, acc: A, fold: &F, leaf: &L) -> A
      where F: Fn(A,&T) -> A,
            L: Fn(usize,&T) -> bool {
      let value = &self.value;
      if leaf(width,value) {
        fold(acc,value)
      }
      else { match self.children {
        None        => fold(acc,value),
        Some(ref x) => {
          let acc = x.0.fold(width/2, acc, fold, leaf);
          let acc = x.1.fold(width/2, acc, fold, leaf);
          let acc = x.2.fold(width/2, acc, fold, leaf);
          let acc = x.3.fold(width/2, acc, fold, leaf);
          acc
        }
      }}
    }
    
    pub fn aggregate<I,L>(&mut self, inner: &I, leaf: &L)
      where I: Fn(&T,&T,&T,&T) -> T,
            L: Fn(&T) -> T {
      self.value = {
        let value    = &self.value;
        let children = &mut self.children;
        children.as_mut().map_or_else(|| leaf(value), |x| {
          x.0.aggregate(inner,leaf);
          x.1.aggregate(inner,leaf);
          x.2.aggregate(inner,leaf);
          x.3.aggregate(inner,leaf);
          inner(&x.0.value, &x.1.value, &x.2.value, &x.3.value)
        })
      };
    }
  }
}

mod hash {
  pub fn pair(k1: usize, k2: usize) -> usize {
    ((k1 + k2) * (k1 + k2 + 1)) / 2 + k2
  }
  
  // taken from:
  // https://web.archive.org/web/20120903003157/http://www.cris.com:80/~Ttwang/tech/inthash.htm    
  pub fn hash(key: i64) -> i64 {
    let key = (!key) + (key << 21);
    let key = key ^ (key >> 24);
    let key = (key + (key << 3)) + (key << 8);
    let key = key ^ (key >> 14);
    let key = (key + (key << 2)) + (key << 4);
    let key = key ^ (key >> 28);
    let key = key + (key << 31);
    key
  }
  
  pub fn noise(x: usize) -> f32 {
    let m = 1 << 16;
    (hash(x as i64).abs() % m) as f32 / (m-1) as f32
  }
}

mod vec2 {
  use std::ops::Add;
  use std::ops::Sub;
  use std::ops::Mul;
  use std::ops::Div;
  
  #[derive(Copy,Clone,Debug)]
  pub struct Vec2<T>(pub T, pub T);

  impl Vec2<f32> {
    pub const ZERO: Vec2<f32> = Vec2(0.0,0.0);
    
    pub fn norm(self) -> f32 {
      (self.0*self.0 + self.1*self.1).sqrt()
    }
    pub fn unit(self) -> Vec2<f32> {
      self / self.norm()
    }
    pub fn dot(self, other: Vec2<f32>) -> f32 {
      self.0 * other.0 + self.1 * other.1
    }
  }
  
  impl<T> Add for Vec2<T>
    where T: Add<Output = T> {
    type Output = Vec2<T>;
    fn add(self, other: Vec2<T>) -> Vec2<T> {
      Vec2(self.0 + other.0, self.1 + other.1)
    }
  }
  impl<T> Add<T> for Vec2<T>
    where T: Add<Output = T> + Copy {
    type Output = Vec2<T>;
    fn add(self, other: T) -> Vec2<T> {
      Vec2(self.0 + other, self.1 + other)
    }
  }
  
  impl<T> Sub for Vec2<T>
    where T: Sub<Output = T> {
    type Output = Vec2<T>;
    fn sub(self, other: Vec2<T>) -> Vec2<T> {
      Vec2(self.0 - other.0, self.1 - other.1)
    }
  }
  impl<T> Sub<T> for Vec2<T>
    where T: Sub<Output = T> + Copy {
    type Output = Vec2<T>;
    fn sub(self, other: T) -> Vec2<T> {
      Vec2(self.0 - other, self.1 - other)
    }
  }
  
  impl<T> Mul for Vec2<T>
    where T: Mul<Output = T> {
    type Output = Vec2<T>;
    fn mul(self, other: Vec2<T>) -> Vec2<T> {
      Vec2(self.0 * other.0, self.1 * other.1)
    }
  }
  impl<T> Mul<T> for Vec2<T>
    where T: Mul<Output = T> + Copy {
    type Output = Vec2<T>;
    fn mul(self, other: T) -> Vec2<T> {
      Vec2(self.0 * other, self.1 * other)
    }
  }
  
  impl<T> Div for Vec2<T>
    where T: Div<Output = T> {
    type Output = Vec2<T>;
    fn div(self, other: Vec2<T>) -> Vec2<T> {
      Vec2(self.0 / other.0, self.1 / other.1)
    }
  }
  impl<T> Div<T> for Vec2<T>
    where T: Div<Output = T> + Copy {
    type Output = Vec2<T>;
    fn div(self, other: T) -> Vec2<T> {
      Vec2(self.0 / other, self.1 / other)
    }
  }
}

mod either {
  pub enum Either<L,R> {Left(L), Right(R)}
}
