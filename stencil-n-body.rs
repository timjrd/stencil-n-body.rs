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
          .args(&["-f", "rawvideo",
                  "-pixel_format", "rgb24",
                  "-video_size", &format!("{}x{}", args.width, args.height),
                  "-i", "-", "-y", args.out])
          .stdin(Stdio::piped())
          .stdout(Stdio::null())
          .stderr(Stdio::null())
          .spawn() {
            Err(_)     => eprintln!("failed to start ffmpeg"),
            Ok(ffmpeg) => if run(args,ffmpeg).is_none() {
              eprintln!("\rio error");
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
      if i % 125 == 0 {grid.reverse_polarity()};
    }
    for i in 0 .. args.it {
      print_it(args.it_before+i);
      grid.iteration();
      if i % 125 == 0 {grid.reverse_polarity()};      
      grid.write_rgb(0.7, 135.0, ffmpeg).ok()?;
      ffmpeg.flush().ok()?;
    }
    println!();
    Some(())
  }
}

mod grid {
  use either::Either;
  use vec2::Vec2;
  use quadtree::Tree;

  #[derive(Copy,Clone)]
  struct Cell {charge: f32, mass: f32}
  #[derive(Copy,Clone)]
  struct Dim {width: usize, height: usize}
  
  struct Node {average: Cell, center: Vec2<f32>, leaf: bool}
  
  pub struct Grid {
    matrix:     Vec<Cell>,
    dim:        Dim,
    tree:       Tree<Node>,
    root_width: usize,
    polarity:   bool
  }
  
  impl Cell {
    const EMPTY: Cell = Cell {charge: 0.0, mass: 0.0};
  }

  impl Dim {
    fn size(self) -> usize {
      self.width * self.height
    }
    fn from(self, i: usize) -> (usize,usize) {
      (i % self.width, i / self.width)
    }
    fn to(self, x: usize, y: usize) -> usize {
      y * self.width + x
    }
    fn ito(self, x: isize, y: isize) -> Option<usize> {
      if x < 0 || y < 0 { None }
      else              { Some(self.to(x as usize, y as usize)) }
    }
  }

  impl Grid {
    const THETA: f32 = 3.0;
    const G:     f32 = 3.0;
    
    pub fn new(width: usize, height: usize) -> Grid {
      let dim = Dim {width: width, height: height};
      let root_width = width.max(height).next_power_of_two();
      
      let matrix: Vec<Cell> = (0..dim.size()).map(|i| {
        let (x,y) = dim.from(i);
        Cell {
          charge: Grid::noise(Grid::pair(x/20, y/20)),
          mass:   1.0
        }
      }).collect();
      
      let tree = {
        let divide = |p| match p {
          (1,x,y) => Either::Right( Node {
            average: *matrix.get(dim.to(x,y)).unwrap_or(&Cell::EMPTY),
            center:  Vec2(x as f32, y as f32),
            leaf:    true
          }),
          (_,x,y) if x >= width || y >= height =>
            Either::Right( Node {
              average: Cell::EMPTY,
              center:  Vec2(x as f32, y as f32),
              leaf:    true
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
        matrix:     matrix,
        dim:        dim,
        tree:       tree,
        root_width: root_width,
        polarity:   true
      }
    }

    pub fn write_rgb<W>(&self, saturation: f32, from_hue: f32, writer: &mut W)
                        -> std::io::Result<()>
      where W: std::io::Write {
      for x in self.matrix.iter() {
        let hue   = (from_hue + x.charge*180.0) % 360.0;
        let value = 1.0 - (1.0 / (x.mass * 4.0 + 1.0));
        
        let f = |n| {
          let k: f32 = (n + hue/60.0) % 6.0;
          value - value * saturation
            * 0_f32.max( 1_f32.min( k.min( 4.0-k )))
        };
        let r = f(5.0) * 255.0;
        let g = f(3.0) * 255.0;
        let b = f(1.0) * 255.0;

        writer.write(&[r as u8, g as u8, b as u8])?;
      }
      Ok(())
    }

    pub fn reverse_polarity(&mut self) {
      self.polarity = ! self.polarity;
    }
    
    pub fn iteration(&mut self) {
      for i in 0 .. self.dim.size() {
        let (x,y) = self.dim.from(i);
        self.kernel(x,y);
      }
      self.update_tree();
    }
    
    fn kernel(&mut self, x: usize, y: usize) {
      let dim  = self.dim;
      if x < dim.width && y < dim.height {
        let polarity = self.polarity;
        let here = Vec2(x as f32, y as f32);      

        let (before,after) = self.matrix.as_mut_slice().split_at_mut(dim.to(x,y));
        let (this,after) = after.split_at_mut(1);
        let this = &mut this[0];

        let net_force = {
          let fold = |acc, node: &Node| {
            let charge_mass = Grid::charge_mass(node.average.charge, this.charge)
              * if polarity {1.0} else {-1.0};
            acc + Grid::force( Grid::G, node.center, here,
                               charge_mass.abs() * node.average.mass,
                               charge_mass.abs() * this.mass ) * charge_mass.signum()
          };
          let leaf = |width, node: &Node| node.leaf
            || width as f32 / (here - node.center).norm() < Grid::THETA;
          
          self.tree.fold(self.root_width, Vec2::ZERO, &fold, &leaf)
        };

        let mut drain = |dx: isize, dy: isize| {
          let dir     = Vec2(dx as f32, dy as f32).unit();
          let angle   = net_force.unit().dot(dir).acos();
          let quarter = std::f32::consts::PI / 4.0;
          if angle < quarter {
            dim.ito(x as isize + dx, y as isize + dy).map(|i| {
              let mut other = if i < dim.to(x,y) {
                before.get_mut(i)
              }
              else {
                after.get_mut(i - dim.to(x,y))
              };
              other.as_mut().map(|other| {
                let drained = ((quarter - angle) / quarter) * net_force.norm();
                let drained = drained.min(this.mass / 4.0);
                if  drained > 0.0 {
                  other.charge = (other.charge*other.mass + this.charge*drained)
                    / (other.mass + drained);
                
                  this.mass  -= drained;
                  other.mass += drained;
                }
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
    }
    
    fn update_tree(&mut self) {
      let matrix = &self.matrix;
      let dim    = self.dim;
      let leaf = |x: &Node| Node {
        average: *matrix.get(dim.to(x.center.0 as usize, x.center.1 as usize))
          .unwrap_or(&Cell::EMPTY), .. *x
      };
      self.tree.aggregate(&Grid::aggregate, &leaf);
    }
    
    fn aggregate(a: &Node, b: &Node, c: &Node, d: &Node) -> Node {
      let mass =
        a.average.mass +
        b.average.mass +
        c.average.mass +
        d.average.mass;
      let (charge,center) = if mass == 0.0 {
        (0.0, (a.center+b.center+c.center+d.center) / 4.0)
      }
      else {(
        ( a.average.charge * a.average.mass +
          b.average.charge * b.average.mass +
          c.average.charge * c.average.mass +
          d.average.charge * d.average.mass ) / mass,
        ( a.center * a.average.mass +
          b.center * b.average.mass +
          c.center * c.average.mass +
          d.center * d.average.mass ) / mass
      )};

      Node {
        average: Cell {charge: charge, mass: mass},
        center:  center,
        leaf:    mass == 0.0
      }
    }
    
    // force applied on object 2 exerted by object 1
    fn force(g: f32, r1: Vec2<f32>, r2: Vec2<f32>, m1: f32, m2: f32) -> Vec2<f32> {
      let r = r2 - r1;
      let d = r.norm();
      if  d == 0.0 { Vec2::ZERO } else {
        r.unit() * (-g) * ((m1 * m2) / (d * d))
      }
    }
    
    // charge_mass > 0 means attraction, repulsion otherwise
    fn charge_mass(c1: f32, c2: f32) -> f32 {
      let x = c2 - c1 - 1.0;
      let x = x*x*x*x * 2.0 - 1.0;
      x*x * 2.0 - 1.0
    }

    // taken from:
    // https://web.archive.org/web/20120903003157/http://www.cris.com:80/~Ttwang/tech/inthash.htm
    fn hash(key: i64) -> i64 {
      let key = (!key) + (key << 21);
      let key = key ^ (key >> 24);
      let key = (key + (key << 3)) + (key << 8);
      let key = key ^ (key >> 14);
      let key = (key + (key << 2)) + (key << 4);
      let key = key ^ (key >> 28);
      let key = key + (key << 31);
      key
    }
    fn noise(x: usize) -> f32 {
      let m = 1 << 16;
      (Grid::hash(x as i64).abs() % m) as f32 / (m-1) as f32
    }
    fn pair(k1: usize, k2: usize) -> usize {
      ((k1 + k2) * (k1 + k2 + 1)) / 2 + k2
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
