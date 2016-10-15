class CreateBabeRuthNumber < ActiveRecord::Migration
  def change
    create_table :babe_ruth_numbers do |t|
      t.string :player_id, null: false
      t.string :team_id, null: false
      t.integer :distance, null: false
    end
    add_index :babe_ruth_numbers, :team_id, unique: true
  end
end
