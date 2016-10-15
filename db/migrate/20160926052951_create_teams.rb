class CreateTeams < ActiveRecord::Migration
  def change
    create_table :teams do |t|
      t.integer :yearid
      t.string :lgid
      t.string :teamid
      t.string :franchid
      t.string :divid
      t.integer :rank
      t.integer :g
      t.integer :ghome
      t.integer :w
      t.integer :l
      t.string :divwin
      t.string :wcwin
      t.string :lgwin
      t.string :wswin
      t.integer :r
      t.integer :ab
      t.integer :h
      t.integer :dub
      t.integer :trip
      t.integer :hr
      t.integer :bb
      t.integer :so
      t.integer :sb
      t.integer :cs
      t.integer :hbp
      t.integer :sf
      t.integer :ra
      t.integer :er
      t.float :era
      t.integer :cg
      t.integer :sho
      t.integer :sv
      t.integer :ipouts
      t.integer :ha
      t.integer :hra
      t.integer :bba
      t.integer :soa
      t.integer :e
      t.integer :dp
      t.float :fp
      t.string :name
      t.string :park
      t.integer :attendance
      t.integer :bpf
      t.integer :ppf
      t.string :teamidbr
      t.string :teamidlahman45
      t.string :teamidretro
    end
  end
end
